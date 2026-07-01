"""Guardrail 4 — live results parse cleanly, degrade offline, and are honored by the simulator.

Phase 1 adds a live-results feed (openfootball, CC0) that locks already-played group matches into
the tournament simulator. These tests are hermetic: parsing is checked on a hand-built payload, the
network path is exercised through an injected downloader (no sockets), and the "results are
honored" check loads the committed wc2026.json (offline) and drives a tiny stub predictor.
"""

from __future__ import annotations

import json
import math

import numpy as np

from core import live, simulate
from core.ingest import load_wc2026


class _StubPredictor:
    """Deterministic strength-based predictor (mirrors the format test's stub); no model needed."""

    def __init__(self, teams, seed: int = 1):
        rng = np.random.default_rng(seed)
        self.strength = {t: float(rng.normal(0, 1)) for t in teams}

    def predict(self, home, away, neutral=True, is_host_home=0):
        s = self.strength[home] - self.strength[away] + (0.0 if neutral else 0.3)
        eH, eA, eD = math.exp(s), math.exp(-s), math.exp(0.0) * 1.3
        z = eH + eA + eD
        return {"H": eH / z, "D": eD / z, "A": eA / z}


# --------------------------------------------------------------------------------------
# Pure parsing
# --------------------------------------------------------------------------------------
def test_parse_openfootball_filters_and_normalizes():
    data = {
        "name": "World Cup 2026",
        "matches": [
            # Played group match in alias spellings -> normalized & included.
            {
                "group": "Group D",
                "team1": "USA",
                "team2": "Czech Republic",
                "score": {"ft": [2, 1]},
            },
            # Unplayed group match (null score) -> excluded.
            {"group": "Group A", "team1": "Mexico", "team2": "South Africa", "score": None},
            # Played knockout match (no group) -> excluded (sim re-draws the bracket).
            {"round": "Round of 32", "team1": "Brazil", "team2": "France", "score": {"ft": [1, 0]}},
            # Group match with only a half-time score -> excluded (not full-time / "played").
            {"group": "Group B", "team1": "Canada", "team2": "Qatar", "score": {"ht": [0, 0]}},
        ],
    }
    results = live.parse_openfootball_matches(data)
    assert results == [
        {"home": "United States", "away": "Czechia", "home_score": 2, "away_score": 1}
    ]


def test_merge_known_results_live_overrides_committed_and_does_not_mutate():
    wc = {"known_results": [{"home": "A", "away": "B", "home_score": 0, "away_score": 0}]}
    live_res = [{"home": "B", "away": "A", "home_score": 3, "away_score": 1}]  # same pair, reversed
    merged = live.merge_known_results(wc, live_res)
    # One entry per unordered pair, with the live result winning.
    assert merged["known_results"] == live_res
    # The input dict is not mutated.
    assert wc["known_results"] == [{"home": "A", "away": "B", "home_score": 0, "away_score": 0}]


# --------------------------------------------------------------------------------------
# Fetch / cache (network injected — no real I/O)
# --------------------------------------------------------------------------------------
def test_fetch_live_results_writes_and_reads_cache(tmp_path):
    cache = tmp_path / "live.json"
    payload = {
        "matches": [
            {
                "group": "Group A",
                "team1": "Mexico",
                "team2": "South Africa",
                "score": {"ft": [2, 0]},
            }
        ]
    }
    snap = live.fetch_live_results(downloader=lambda url, timeout: payload, cache_path=cache)
    assert snap["known_results"] == [
        {"home": "Mexico", "away": "South Africa", "home_score": 2, "away_score": 0}
    ]
    assert snap["fetched_at"] is not None
    assert cache.exists()
    assert live.read_cache(cache)["known_results"] == snap["known_results"]


def test_fetch_live_results_falls_back_to_cache_offline(tmp_path):
    cache = tmp_path / "wc2026_live.json"
    cached = {
        "fetched_at": "2026-06-15T12:00:00+00:00",
        "source": "openfootball/worldcup.json",
        "url": "https://example/worldcup.json",
        "known_results": [
            {"home": "Mexico", "away": "South Africa", "home_score": 2, "away_score": 0}
        ],
    }
    cache.write_text(json.dumps(cached), encoding="utf-8")

    def _boom(url, timeout):
        raise OSError("network down")

    snap = live.fetch_live_results(downloader=_boom, cache_path=cache)
    # Last good snapshot is reused, with its original timestamp, marked as cached.
    assert snap["known_results"] == cached["known_results"]
    assert snap["fetched_at"] == "2026-06-15T12:00:00+00:00"
    assert "cached" in snap["source"]


def test_fetch_live_results_empty_snapshot_when_offline_and_no_cache(tmp_path):
    def _boom(url, timeout):
        raise OSError("network down")

    snap = live.fetch_live_results(downloader=_boom, cache_path=tmp_path / "missing.json")
    assert snap["known_results"] == []
    assert snap["fetched_at"] is None
    assert "error" in snap  # offline state is reported, not hidden


# --------------------------------------------------------------------------------------
# Locked results are honored by the simulator, and they change the standings
# --------------------------------------------------------------------------------------
def test_locked_results_are_honored_and_standings_update():
    wc = load_wc2026()  # committed file, no network
    teams = [t for grp in wc["groups"].values() for t in grp]
    stub = _StubPredictor(teams, seed=1)
    n_sims = 300

    # Force a full, transitive Group A table that contradicts the stub's strengths: the team the
    # stub rates WEAKEST wins all three (9 pts -> 1st), the strongest loses all three (0 pts -> 4th).
    group_a = wc["groups"]["A"]
    r0, r1, r2, r3 = sorted(group_a, key=lambda t: stub.strength[t])  # r0 weakest, r3 strongest
    locked = [
        {"home": r0, "away": r1, "home_score": 1, "away_score": 0},
        {"home": r0, "away": r2, "home_score": 1, "away_score": 0},
        {"home": r0, "away": r3, "home_score": 1, "away_score": 0},
        {"home": r1, "away": r2, "home_score": 1, "away_score": 0},
        {"home": r1, "away": r3, "home_score": 1, "away_score": 0},
        {"home": r2, "away": r3, "home_score": 1, "away_score": 0},
    ]

    base = simulate.TournamentSimulator(stub, wc, seed=0).run(n_sims=n_sims, seed=0)
    base_r32 = base.table.set_index("team")["R32"]

    wc_locked = live.merge_known_results(wc, locked)
    locked_res = simulate.TournamentSimulator(stub, wc_locked, seed=0).run(n_sims=n_sims, seed=0)
    locked_r32 = locked_res.table.set_index("team")["R32"]

    # Honored: the forced winner & runner-up ALWAYS reach the R32; the forced last-placed team
    # (0 pts) NEVER does — 4th place cannot qualify, even as a best third.
    assert locked_r32[r0] == 1.0
    assert locked_r32[r1] == 1.0
    assert locked_r32[r3] == 0.0

    # Standings genuinely updated vs the unlocked sim: the on-paper-weakest team was not a sure
    # qualifier before and now is; the on-paper-strongest was a likely qualifier and is now out.
    assert base_r32[r0] < 1.0
    assert locked_r32[r0] > base_r32[r0]
    assert locked_r32[r3] < base_r32[r3]


# --------------------------------------------------------------------------------------
# Knockout results: parse winners (incl. ET / penalties), merge, and are honored by the sim
# --------------------------------------------------------------------------------------
def test_parse_openfootball_knockout_ft_et_pens_and_skips():
    data = {
        "matches": [
            # Group match -> excluded (knockouts only here; groups go through the other parser).
            {
                "group": "Group A",
                "team1": "Mexico",
                "team2": "South Africa",
                "score": {"ft": [2, 0]},
            },
            # Decided in regulation.
            {"round": "Round of 32", "team1": "Brazil", "team2": "Japan", "score": {"ft": [2, 1]}},
            # Level at full time, decided in extra time.
            {
                "round": "Round of 32",
                "team1": "Spain",
                "team2": "Italy",
                "score": {"ft": [1, 1], "et": [2, 1]},
            },
            # Level through ET, decided on penalties -> winner from the shootout (real 2026 tie).
            {
                "round": "Round of 32",
                "team1": "Netherlands",
                "team2": "Morocco",
                "score": {"p": [2, 3], "et": [1, 1], "ft": [1, 1], "ht": [0, 0]},
            },
            # Still level with no ET/shootout data -> undecided -> skipped.
            {"round": "Round of 16", "team1": "France", "team2": "Sweden", "score": {"ft": [1, 1]}},
            # Not played -> skipped.
            {"round": "Round of 16", "team1": "England", "team2": "Portugal", "score": None},
        ]
    }
    got = {
        (r["home"], r["away"]): (r["winner"], r["decided_by"], r["home_score"], r["away_score"])
        for r in live.parse_openfootball_knockout(data)
    }
    assert got == {
        ("Brazil", "Japan"): ("Brazil", "ft", 2, 1),
        ("Spain", "Italy"): ("Spain", "et", 1, 1),
        # Netherlands lost the shootout despite a level score line — the winner, not the score, is
        # what gets locked; home/away scores keep the full-time 1–1 for display.
        ("Netherlands", "Morocco"): ("Morocco", "pen", 1, 1),
    }


def test_merge_known_ko_results_overrides_and_no_mutation():
    wc = {
        "known_ko_results": [
            {
                "home": "A",
                "away": "B",
                "home_score": 1,
                "away_score": 1,
                "winner": "A",
                "decided_by": "ft",
            }
        ]
    }
    live_ko = [  # same pair, reversed orientation, live flips the winner
        {
            "home": "B",
            "away": "A",
            "home_score": 0,
            "away_score": 0,
            "winner": "B",
            "decided_by": "pen",
        }
    ]
    merged = live.merge_known_ko_results(wc, live_ko)
    assert merged["known_ko_results"] == live_ko  # one entry per unordered pair, live wins
    assert wc["known_ko_results"][0]["winner"] == "A"  # input not mutated


def _lock_full_group_stage(wc: dict) -> list[dict]:
    """A deterministic 9/6/3/0 finish in every group (listed order), as ``known_results`` rows."""
    order = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]  # each earlier-listed team beats later
    locked = []
    for teams in wc["groups"].values():
        for i, j in order:
            locked.append({"home": teams[i], "away": teams[j], "home_score": 1, "away_score": 0})
    return locked


def test_ko_result_honored_by_simulator():
    wc = load_wc2026()
    teams = [t for grp in wc["groups"].values() for t in grp]
    stub = _StubPredictor(teams, seed=1)
    # Distinct FIFA ranks break the (identical) best-thirds records deterministically, so with the
    # full group stage locked the entire R32 bracket is fixed across sims — the precondition that
    # lets us assert exact 0.0/1.0 probabilities.
    fifa = {t: i + 1 for i, t in enumerate(teams)}
    wc_g = live.merge_known_results(wc, _lock_full_group_stage(wc))

    probe = simulate.TournamentSimulator(stub, wc_g, seed=0, fifa_rank=fifa)
    probe.simulate_once()
    a, b = probe._last_r32_pairs[0]  # a real, fixed R32 tie
    underdog = a if stub.strength[a] < stub.strength[b] else b  # the side the model rates weaker
    favorite = b if underdog == a else a

    ko = [
        {
            "home": a,
            "away": b,
            "home_score": 0,
            "away_score": 0,
            "winner": underdog,
            "decided_by": "pen",
        }
    ]
    wc_ko = live.merge_known_ko_results(wc_g, ko)
    res = simulate.TournamentSimulator(stub, wc_ko, seed=0, fifa_rank=fifa).run(n_sims=200, seed=0)
    r16 = res.table.set_index("team")["R16"]

    assert r16[underdog] == 1.0  # forced winner always advances out of the R32
    assert (
        r16[favorite] == 0.0
    )  # forced loser never reaches the R16, model strength notwithstanding


def test_eliminated_team_never_advances_under_varying_bracket():
    """A team knocked out for real never advances — even where the re-drawn bracket varies."""
    wc = load_wc2026()
    teams = [t for grp in wc["groups"].values() for t in grp]
    stub = _StubPredictor(teams, seed=2)

    # Lock ONLY Group A (so its winner always makes the R32); every other group is sampled, so this
    # team meets a different R32 opponent from sim to sim — exercising the eliminated-set guard, not
    # just the exact-pairing lock.
    group_a = wc["groups"]["A"]
    x = group_a[0]  # Group A winner under a 9/6/3/0 lock below
    order = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    group_a_locked = [
        {"home": group_a[i], "away": group_a[j], "home_score": 1, "away_score": 0} for i, j in order
    ]
    wc_g = live.merge_known_results(wc, group_a_locked)

    # Record X as having lost a knockout tie (opponent from another group) -> X is eliminated.
    opponent = wc["groups"]["B"][0]
    ko = [
        {
            "home": x,
            "away": opponent,
            "home_score": 1,
            "away_score": 1,
            "winner": opponent,
            "decided_by": "pen",
        }
    ]
    wc_ko = live.merge_known_ko_results(wc_g, ko)

    res = simulate.TournamentSimulator(stub, wc_ko, seed=0, fifa_rank=None).run(n_sims=300, seed=0)
    row = res.table.set_index("team")
    assert row.loc[x, "R32"] == 1.0  # still qualifies (locked Group A winner)
    assert row.loc[x, "R16"] == 0.0  # but never wins a knockout tie
    assert row.loc[x, "SF"] == 0.0
    assert row.loc[x, "Champion"] == 0.0
