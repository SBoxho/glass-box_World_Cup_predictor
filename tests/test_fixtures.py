"""Guardrail 5 — the schedule provider parses cleanly, normalizes, and degrades offline.

The Matchday Home page is driven by :mod:`core.fixtures`, which projects the same openfootball feed
into a full per-fixture schema (kickoff in UTC, venue, status). These tests are hermetic: parsing
and kickoff/timezone math run on hand-built payloads, selection (live / next / today) runs against
a frozen ``now``, and the network path is exercised through an injected downloader (no sockets).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta, timezone

from core import fixtures as fx


# --------------------------------------------------------------------------------------
# Kickoff parsing (date + "HH:MM UTC±H[:MM]") → absolute UTC instant
# --------------------------------------------------------------------------------------
def test_parse_kickoff_applies_utc_offset():
    # "13:00 UTC-6" on 11 June is 19:00 UTC.
    ko = fx.parse_kickoff("2026-06-11", "13:00 UTC-6")
    assert ko == datetime(2026, 6, 11, 19, 0, tzinfo=UTC)


def test_parse_kickoff_handles_half_hour_offsets_both_signs():
    assert fx.parse_kickoff("2026-06-11", "20:00 UTC+5:30") == datetime(
        2026, 6, 11, 14, 30, tzinfo=UTC
    )
    assert fx.parse_kickoff("2026-06-11", "18:00 UTC-3:30") == datetime(
        2026, 6, 11, 21, 30, tzinfo=UTC
    )


def test_parse_kickoff_assumes_utc_without_offset_and_fails_soft():
    assert fx.parse_kickoff("2026-06-11", "18:00") == datetime(2026, 6, 11, 18, 0, tzinfo=UTC)
    assert fx.parse_kickoff("2026-06-11", None) is None  # date but no time → unknown kickoff
    assert fx.parse_kickoff(None, "18:00") is None
    assert fx.parse_kickoff("2026-06-11", "garbage") is None


# --------------------------------------------------------------------------------------
# Pure parsing of the feed payload
# --------------------------------------------------------------------------------------
def _payload() -> dict:
    return {
        "name": "World Cup 2026",
        "matches": [
            {  # played group match, alias spellings -> normalized; scores align to team1/team2
                "round": "Matchday 1",
                "date": "2026-06-11",
                "time": "13:00 UTC-6",
                "team1": "USA",
                "team2": "Czech Republic",
                "score": {"ft": [2, 1]},
                "goals1": [{"name": "Scorer A", "minute": "9"}],
                "goals2": [],
                "group": "Group D",
                "ground": "Dallas",
            },
            {  # unplayed group match -> kept, scores None
                "round": "Matchday 8",
                "date": "2026-06-18",
                "time": "12:00 UTC-4",
                "team1": "Mexico",
                "team2": "South Africa",
                "group": "Group A",
                "ground": "Atlanta",
            },
            {  # knockout match -> kept, group is None
                "round": "Round of 32",
                "date": "2026-06-29",
                "time": "16:00 UTC-5",
                "team1": "Brazil",
                "team2": "France",
                "ground": "New York",
            },
        ],
    }


def test_parse_fixtures_keeps_all_matches_and_normalizes():
    out = fx.parse_fixtures(_payload())
    assert len(out) == 3  # unlike core.live, upcoming + knockout fixtures are NOT dropped

    played = out[0]
    assert played["team1"] == "United States" and played["team2"] == "Czechia"
    assert played["group"] == "D"
    assert played["finished"] is True
    assert (played["team1_score"], played["team2_score"]) == (2, 1)
    assert played["venue"] == "Dallas"
    assert played["kickoff_utc"] == "2026-06-11T19:00:00+00:00"
    assert played["scorers1"] == ["Scorer A 9'"]

    upcoming = out[1]
    assert upcoming["finished"] is False
    assert upcoming["team1_score"] is None and upcoming["team2_score"] is None

    knockout = out[2]
    assert knockout["group"] is None  # no group field → neutral-venue tie downstream
    assert knockout["round"] == "Round of 32"


def test_parse_fixtures_skips_rows_missing_a_team():
    out = fx.parse_fixtures({"matches": [{"team1": "Brazil", "team2": "", "group": "Group C"}]})
    assert out == []


# --------------------------------------------------------------------------------------
# Status classification + selection (frozen `now`, no clock dependence)
# --------------------------------------------------------------------------------------
def _fixtures_for_selection() -> list[dict]:
    return fx.parse_fixtures(
        {
            "matches": [
                {  # finished earlier today
                    "date": "2026-06-18",
                    "time": "02:00 UTC",
                    "team1": "Iraq",
                    "team2": "Norway",
                    "group": "Group I",
                    "score": {"ft": [1, 0]},
                },
                {  # kicked off 60' ago, no score yet -> live
                    "date": "2026-06-18",
                    "time": "17:00 UTC",
                    "team1": "Spain",
                    "team2": "Brazil",
                    "group": "Group H",
                },
                {  # kicks off in 2h -> upcoming (and today, in UTC)
                    "date": "2026-06-18",
                    "time": "20:00 UTC",
                    "team1": "France",
                    "team2": "Senegal",
                    "group": "Group I",
                },
                {  # tomorrow -> upcoming but not today
                    "date": "2026-06-19",
                    "time": "16:00 UTC",
                    "team1": "England",
                    "team2": "Croatia",
                    "group": "Group L",
                },
                {  # date only, no time -> scheduled, not orderable
                    "date": "2026-06-18",
                    "team1": "Ghana",
                    "team2": "Panama",
                    "group": "Group L",
                },
            ]
        }
    )


def test_classify_status_buckets():
    now = datetime(2026, 6, 18, 18, 0, tzinfo=UTC)
    F = _fixtures_for_selection()
    statuses = {(f["team1"], f["team2"]): fx.classify_status(f, now) for f in F}
    assert statuses[("Iraq", "Norway")] == "finished"
    assert statuses[("Spain", "Brazil")] == "live"
    assert statuses[("France", "Senegal")] == "upcoming"
    assert statuses[("England", "Croatia")] == "upcoming"
    assert statuses[("Ghana", "Panama")] == "scheduled"


def test_classify_status_recent_after_live_window():
    # 3 hours after kickoff with still no score -> past the live window -> "recent".
    f = fx.parse_fixtures(
        {"matches": [{"date": "2026-06-18", "time": "12:00 UTC", "team1": "A", "team2": "B"}]}
    )[0]
    assert fx.classify_status(f, datetime(2026, 6, 18, 15, 0, tzinfo=UTC)) == "recent"


def test_current_and_next_match_selection():
    now = datetime(2026, 6, 18, 18, 0, tzinfo=UTC)
    F = _fixtures_for_selection()
    live = fx.current_match(F, now)
    assert live is not None and (live["team1"], live["team2"]) == ("Spain", "Brazil")
    nxt = fx.next_match(F, now)
    assert nxt is not None and (nxt["team1"], nxt["team2"]) == ("France", "Senegal")
    assert fx.has_remaining_matches(F, now) is True


def test_next_match_none_when_tournament_over():
    now = datetime(2026, 8, 1, 0, 0, tzinfo=UTC)  # well after every fixture
    F = _fixtures_for_selection()
    assert fx.current_match(F, now) is None
    assert fx.next_match(F, now) is None
    assert fx.has_remaining_matches(F, now) is False


def test_todays_matches_respects_timezone():
    F = _fixtures_for_selection()
    now = datetime(2026, 6, 18, 18, 0, tzinfo=UTC)

    today_utc = fx.todays_matches(F, now, UTC)
    pairs = [(f["team1"], f["team2"]) for f in today_utc]
    # Three timed today + one date-only today; tomorrow's match excluded. Timed come first, sorted.
    assert ("England", "Croatia") not in pairs
    assert pairs[:3] == [("Iraq", "Norway"), ("Spain", "Brazil"), ("France", "Senegal")]
    assert ("Ghana", "Panama") in pairs  # date-only fixture appended

    # In UTC+10 the local "today" is already 19 June (now is 04:00 the next day), so the 02:00Z
    # match (12:00 on the 18th locally) falls off "today" — proof the zone shifts the day boundary.
    tz_plus10 = timezone(timedelta(hours=10))
    today_local = [(f["team1"], f["team2"]) for f in fx.todays_matches(F, now, tz_plus10)]
    assert ("Iraq", "Norway") not in today_local
    assert ("Spain", "Brazil") in today_local


# --------------------------------------------------------------------------------------
# Fetch / cache (network injected — no real I/O), mirrors core.live's contract
# --------------------------------------------------------------------------------------
def test_fetch_fixtures_writes_and_reads_cache(tmp_path):
    cache = tmp_path / "fixtures.json"
    snap = fx.fetch_fixtures(downloader=lambda url, timeout: _payload(), cache_path=cache)
    assert len(snap["fixtures"]) == 3
    assert snap["fetched_at"] is not None
    assert cache.exists()
    assert fx.read_cache(cache)["fixtures"][0]["team1"] == "United States"


def test_fetch_fixtures_falls_back_to_cache_offline(tmp_path):
    cache = tmp_path / "fixtures.json"
    cached = {
        "fetched_at": "2026-06-18T10:00:00+00:00",
        "source": "openfootball/worldcup.json",
        "url": "https://example/worldcup.json",
        "fixtures": [{"team1": "Mexico", "team2": "South Africa"}],
    }
    cache.write_text(json.dumps(cached), encoding="utf-8")

    def _boom(url, timeout):
        raise OSError("network down")

    snap = fx.fetch_fixtures(downloader=_boom, cache_path=cache)
    assert snap["fixtures"] == cached["fixtures"]
    assert snap["fetched_at"] == "2026-06-18T10:00:00+00:00"
    assert "cached" in snap["source"]


def test_fetch_fixtures_empty_snapshot_when_offline_and_no_cache(tmp_path):
    def _boom(url, timeout):
        raise OSError("network down")

    snap = fx.fetch_fixtures(downloader=_boom, cache_path=tmp_path / "missing.json")
    assert snap["fixtures"] == []
    assert snap["fetched_at"] is None
    assert "error" in snap  # offline state reported, not hidden


# --------------------------------------------------------------------------------------
# Timezone resolution (browser zone → Europe/Luxembourg → UTC)
# --------------------------------------------------------------------------------------
def test_resolve_timezone_uses_browser_zone_then_falls_back_to_luxembourg():
    assert fx.resolve_timezone("America/New_York")[1] == "America/New_York"
    assert fx.resolve_timezone(None)[1] == "Europe/Luxembourg"  # unknown browser zone
    assert fx.resolve_timezone("")[1] == "Europe/Luxembourg"
    assert fx.resolve_timezone("Not/AZone")[1] == "Europe/Luxembourg"  # unresolvable → default


# --------------------------------------------------------------------------------------
# Committed schedule loader + live-results overlay (the UI-safe resolver)
# --------------------------------------------------------------------------------------
def test_load_committed_fixtures_normalizes_skips_and_defaults_unplayed():
    out = fx.load_committed_fixtures(
        {
            "fixtures": [
                {  # alias spellings → normalized; structure only, no scores in the committed block
                    "team1": "USA",
                    "team2": "Czech Republic",
                    "group": "D",
                    "round": "Matchday 1",
                    "venue": "Dallas",
                    "date": "2026-06-11",
                    "kickoff_utc": "2026-06-11T19:00:00+00:00",
                },
                {"team1": "Brazil", "team2": "", "group": "C"},  # missing a team → skipped
            ]
        }
    )
    assert len(out) == 1
    f = out[0]
    assert (f["team1"], f["team2"]) == ("United States", "Czechia")
    assert f["finished"] is False
    assert f["team1_score"] is None and f["team2_score"] is None
    assert f["scorers1"] == [] and f["scorers2"] == []
    assert f["kickoff_utc"] == "2026-06-11T19:00:00+00:00"


def test_merge_results_into_schedule_orients_scores_and_filters():
    schedule = fx.load_committed_fixtures(
        {
            "fixtures": [
                {"team1": "Spain", "team2": "Brazil", "kickoff_utc": "2026-06-18T17:00:00+00:00"},
                {"team1": "France", "team2": "Senegal", "kickoff_utc": "2026-06-18T20:00:00+00:00"},
            ]
        }
    )
    merged = fx.merge_results_into_schedule(
        schedule,
        [
            {
                "home": "Brazil",
                "away": "Spain",
                "home_score": 2,
                "away_score": 1,
            },  # swapped vs team1/2
            {
                "home": "Ghana",
                "away": "Panama",
                "home_score": 3,
                "away_score": 0,
            },  # no fixture → ignored
            {
                "home": "France",
                "away": "Senegal",
                "home_score": None,
                "away_score": 1,
            },  # null → ignored
        ],
    )
    spain = merged[0]
    assert spain["finished"] is True
    # Brazil (home) scored 2, Spain (away) 1; oriented to the fixture's team1=Spain, team2=Brazil:
    assert (spain["team1_score"], spain["team2_score"]) == (1, 2)
    assert merged[1]["finished"] is False  # null score never marks a match finished


def test_resolve_fixtures_overlays_committed_then_cache_overrides(tmp_path):
    wc = {
        "fixtures": [
            {"team1": "Mexico", "team2": "South Africa", "kickoff_utc": "2026-06-11T19:00:00+00:00"}
        ],
        "known_results": [
            {"home": "Mexico", "away": "South Africa", "home_score": 1, "away_score": 1}
        ],
    }
    # Committed result applies when the live cache is off.
    res = fx.resolve_fixtures(wc=wc, include_live_cache=False)
    assert (res[0]["team1_score"], res[0]["team2_score"], res[0]["finished"]) == (1, 1, True)

    # A fresher cached result overrides the committed one.
    cache = tmp_path / "wc2026_live.json"
    cache.write_text(
        json.dumps(
            {
                "fetched_at": "2026-06-11T22:00:00+00:00",
                "known_results": [
                    {"home": "Mexico", "away": "South Africa", "home_score": 2, "away_score": 0}
                ],
            }
        ),
        encoding="utf-8",
    )
    res2 = fx.resolve_fixtures(wc=wc, include_live_cache=True, live_cache_path=cache)
    assert (res2[0]["team1_score"], res2[0]["team2_score"]) == (2, 0)


def test_committed_schedule_resolves_offline():
    """Guardrail: the committed data/wc2026.json schedule is present, complete and parseable with
    no network and no gitignored cache — so the home page can always decide what to show."""
    resolved = fx.resolve_fixtures(include_live_cache=False)
    assert len(resolved) >= 64  # at least the full group stage is committed
    assert all(f["kickoff_utc"] for f in resolved)  # every fixture has a resolved UTC kick-off
    ko = fx.kickoff_datetime(resolved[0])
    assert ko is not None and ko.tzinfo is not None  # timezone-aware


# --------------------------------------------------------------------------------------
# Coarse status, recency window, and countdown formatting
# --------------------------------------------------------------------------------------
def test_match_status_coarsens_to_four_buckets():
    now = datetime(2026, 6, 18, 18, 0, tzinfo=UTC)  # 6h after the 12:00 kick-off below
    base = {"team1": "A", "team2": "B", "scorers1": [], "scorers2": []}
    upcoming = {**base, "kickoff_utc": "2026-06-18T20:00:00+00:00", "finished": False}
    live = {**base, "kickoff_utc": "2026-06-18T17:00:00+00:00", "finished": False}
    finished = {
        **base,
        "kickoff_utc": "2026-06-18T12:00:00+00:00",
        "finished": True,
        "team1_score": 1,
        "team2_score": 0,
    }
    recent = {
        **base,
        "kickoff_utc": "2026-06-18T12:00:00+00:00",
        "finished": False,
    }  # past window, no score
    scheduled = {**base, "kickoff_utc": None, "finished": False}  # no time → unorderable
    assert fx.match_status(upcoming, now) == "upcoming"
    assert fx.match_status(live, now) == "live"
    assert fx.match_status(finished, now) == "finished"
    assert fx.match_status(recent, now) == "unknown"  # classify_status "recent" coarsens to unknown
    assert fx.match_status(scheduled, now) == "unknown"  # classify_status "scheduled" too


def test_format_time_until():
    assert fx.format_time_until(0) == "now"
    assert fx.format_time_until(-10) == "now"
    assert fx.format_time_until(30) == "<1m"
    assert fx.format_time_until(7 * 60) == "7m"
    assert fx.format_time_until(timedelta(hours=2, minutes=15)) == "2h 15m"
    assert fx.format_time_until(timedelta(days=3, hours=4)) == "3d 4h"


def _at(now: datetime, offset_h: float, *, played: bool) -> dict:
    ko = (now - timedelta(hours=offset_h)).isoformat()
    return {
        "team1": f"H{offset_h}",
        "team2": f"A{offset_h}",
        "group": None,
        "round": "R",
        "venue": "V",
        "date": None,
        "kickoff_utc": ko,
        "finished": played,
        "team1_score": 1 if played else None,
        "team2_score": 0 if played else None,
        "scorers1": [],
        "scorers2": [],
    }


def test_recently_finished_respects_window_order_and_limit():
    now = datetime(2026, 6, 18, 18, 0, tzinfo=UTC)
    fixtures = [
        _at(now, 1, played=True),
        _at(now, 10, played=True),
        _at(now, 60, played=True),
        _at(now, 2, played=False),
    ]
    recent = fx.recently_finished(fixtures, now, within=timedelta(hours=48))
    assert [f["team1"] for f in recent] == [
        "H1",
        "H10",
    ]  # newest first; 60h-old + unplayed excluded
    assert [f["team1"] for f in fx.recently_finished(fixtures, now, limit=1)] == ["H1"]


# --------------------------------------------------------------------------------------
# Match lifecycle + matchday context (before / during / after — frozen datetimes, no clock)
# --------------------------------------------------------------------------------------
def test_match_lifecycle_before_during_after():
    schedule = fx.load_committed_fixtures(
        {
            "fixtures": [
                {"team1": "Spain", "team2": "Brazil", "kickoff_utc": "2026-06-18T17:00:00+00:00"}
            ]
        }
    )
    unplayed = schedule[0]
    before = datetime(2026, 6, 18, 15, 0, tzinfo=UTC)
    during = datetime(2026, 6, 18, 18, 0, tzinfo=UTC)  # 60' after kick-off
    after = datetime(2026, 6, 18, 21, 0, tzinfo=UTC)

    assert fx.match_status(unplayed, before) == "upcoming"
    assert fx.match_status(unplayed, during) == "live"
    assert fx.get_live_matches(during, UTC, fixtures=schedule) == [unplayed]
    assert fx.get_next_match(before, UTC, fixtures=schedule) == unplayed

    played = fx.merge_results_into_schedule(
        schedule, [{"home": "Spain", "away": "Brazil", "home_score": 1, "away_score": 0}]
    )[0]
    assert fx.match_status(played, after) == "finished"


def _matchday_fixtures(with_results: bool):
    wc = {
        "fixtures": [
            {
                "team1": "Iraq",
                "team2": "Norway",
                "group": "I",
                "round": "Matchday 7",
                "venue": "Seattle",
                "date": "2026-06-18",
                "kickoff_utc": "2026-06-18T02:00:00+00:00",
            },
            {
                "team1": "Spain",
                "team2": "Brazil",
                "group": "H",
                "round": "Matchday 7",
                "venue": "Dallas",
                "date": "2026-06-18",
                "kickoff_utc": "2026-06-18T17:00:00+00:00",
            },
            {
                "team1": "France",
                "team2": "Senegal",
                "group": "I",
                "round": "Matchday 7",
                "venue": "Atlanta",
                "date": "2026-06-18",
                "kickoff_utc": "2026-06-18T20:00:00+00:00",
            },
            {
                "team1": "England",
                "team2": "Croatia",
                "group": "L",
                "round": "Matchday 7",
                "venue": "Boston",
                "date": "2026-06-19",
                "kickoff_utc": "2026-06-19T16:00:00+00:00",
            },
        ],
        "known_results": (
            [{"home": "Iraq", "away": "Norway", "home_score": 1, "away_score": 0}]
            if with_results
            else []
        ),
    }
    return fx.resolve_fixtures(wc=wc, include_live_cache=False)


def test_matchday_context_pre_tournament_counts_down_to_first_match():
    fixtures = _matchday_fixtures(with_results=False)
    ctx = fx.get_matchday_context(datetime(2026, 6, 10, 0, 0, tzinfo=UTC), UTC, fixtures=fixtures)
    assert ctx.headline == "next"
    assert ctx.live == []
    assert (ctx.next_match["team1"], ctx.next_match["team2"]) == (
        "Iraq",
        "Norway",
    )  # earliest kick-off
    assert ctx.recently_finished == []
    assert ctx.today == []  # nothing on 10 Jun
    assert ctx.tournament_over is False


def test_matchday_context_mid_tournament_leads_with_live():
    now = datetime(2026, 6, 18, 18, 0, tzinfo=UTC)
    ctx = fx.get_matchday_context(now, UTC, fixtures=_matchday_fixtures(with_results=True))
    assert ctx.headline == "live"
    assert [(f["team1"], f["team2"]) for f in ctx.live] == [("Spain", "Brazil")]
    assert (ctx.next_match["team1"], ctx.next_match["team2"]) == ("France", "Senegal")
    assert ctx.next_in == "2h 0m"
    # today (UTC): the three 18 Jun matches in kick-off order; tomorrow's England match excluded.
    assert [(f["team1"], f["team2"]) for f in ctx.today] == [
        ("Iraq", "Norway"),
        ("Spain", "Brazil"),
        ("France", "Senegal"),
    ]
    assert [(f["team1"], f["team2"]) for f in ctx.recently_finished] == [("Iraq", "Norway")]
    assert ctx.tournament_over is False


def test_matchday_context_post_tournament_is_over():
    wc = {
        "fixtures": _matchday_fixtures(with_results=False),
        "known_results": [
            {"home": "Iraq", "away": "Norway", "home_score": 1, "away_score": 0},
            {"home": "Spain", "away": "Brazil", "home_score": 2, "away_score": 2},
            {"home": "France", "away": "Senegal", "home_score": 3, "away_score": 1},
            {"home": "England", "away": "Croatia", "home_score": 0, "away_score": 0},
        ],
    }
    # resolve_fixtures expects committed `fixtures` as raw schedule rows; reuse the resolved list as
    # the schedule (load_committed_fixtures re-reads team1/team2/kickoff_utc and re-defaults scores).
    fixtures = fx.resolve_fixtures(wc=wc, include_live_cache=False)
    ctx = fx.get_matchday_context(datetime(2026, 8, 1, 0, 0, tzinfo=UTC), UTC, fixtures=fixtures)
    assert ctx.live == []
    assert ctx.next_match is None
    assert ctx.tournament_over is True
    assert ctx.headline == "over"
    assert ctx.recently_finished == []  # everything is well outside the 48h window


def test_get_today_matches_shifts_with_timezone():
    wc = {
        "fixtures": [
            {
                "team1": "Iraq",
                "team2": "Norway",
                "date": "2026-06-18",
                "kickoff_utc": "2026-06-18T02:00:00+00:00",
            },
            {
                "team1": "Spain",
                "team2": "Brazil",
                "date": "2026-06-18",
                "kickoff_utc": "2026-06-18T17:00:00+00:00",
            },
        ]
    }
    fixtures = fx.resolve_fixtures(wc=wc, include_live_cache=False)
    now = datetime(2026, 6, 18, 18, 0, tzinfo=UTC)

    assert [f["team1"] for f in fx.get_today_matches(now, UTC, fixtures=fixtures)] == [
        "Iraq",
        "Spain",
    ]

    # In UTC+10 the local clock already reads 19 June 04:00: the 02:00Z match (12:00 on the 18th
    # locally) drops off "today", while the 17:00Z match (03:00 on the 19th locally) is today.
    tz_plus10 = timezone(timedelta(hours=10))
    today_p10 = [f["team1"] for f in fx.get_today_matches(now, tz_plus10, fixtures=fixtures)]
    assert "Iraq" not in today_p10 and "Spain" in today_p10
