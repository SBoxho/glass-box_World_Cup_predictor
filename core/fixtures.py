"""2026 World Cup schedule — the full fixture list (kickoffs, venues, status), for the home page.

:mod:`core.live` already reads the openfootball feed, but it keeps only *played group* results
(the simulator's ``known_results`` block) and discards everything else — dates, kickoff times,
venues, upcoming fixtures. The "Matchday Home" landing page needs exactly what ``live`` throws
away: *when* the next match kicks off, *where*, and what is on *today*. This module parses the
same public feed into a richer per-fixture schema and answers "what's live / next / today".

Design mirrors :mod:`core.live` so the two stay consistent:

* **Framework-free.** No Streamlit/FastAPI imports — the app layer adds caching and the UI.
* **Pure parse split from I/O.** :func:`parse_fixtures` / :func:`parse_kickoff` are pure
  (unit-tested without a network); :func:`fetch_fixtures` does the download behind an injectable
  ``downloader`` and degrades to the last cached snapshot offline.
* **Time is normalized to UTC.** The feed stamps each match like ``"13:00 UTC-6"`` (host cities
  span several time zones); we resolve every kickoff to an absolute UTC instant so the app can
  render it in the *user's* local zone and reason about "live / upcoming" against ``now``.
* **No live in-play feed.** openfootball is a results feed, not a minute-by-minute ticker, so a
  match is inferred "live" when it has kicked off, has no full-time score yet, and is still inside
  a plausible match window. The UI labels any derived minute as approximate (honest, not faked).

The page's *default* data path is committed-first and offline, not the network: :func:`build_schedule`
bakes a structure-only schedule into ``data/wc2026.json`` (``fixtures[]`` — kickoffs/venues, no
scores), and :func:`resolve_fixtures` loads it and overlays played results (committed
``known_results`` + the cached live snapshot) on top. The public matchday API —
:func:`get_live_matches`, :func:`get_next_match`, :func:`get_today_matches` and the one-call
:func:`get_matchday_context` — takes ``(now, timezone)`` and returns render-ready objects, with
:func:`resolve_timezone` resolving the browser zone (fallback Europe/Luxembourg → UTC). All of this
is network-free, so the home page can *always* decide what to show; :func:`fetch_fixtures` (the live
feed) is only pulled on an explicit Refresh.

Source: ``openfootball/worldcup.json`` (CC0 / public domain, no API key). Reuses the same URL as
:mod:`core.live` (:data:`core.config.WC2026_LIVE_URL`). See ``DATA_SOURCES.md``.
"""

from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone, tzinfo
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from core import config

SOURCE = "openfootball/worldcup.json"

# User-timezone fallback when the browser/local zone is unknown or unresolvable. The app's author
# and primary audience are in Luxembourg, so this is the most useful default for "local kick-off
# times" — it degrades to UTC only if the tz database itself is missing. See :func:`resolve_timezone`.
DEFAULT_TIMEZONE = "Europe/Luxembourg"

# Cached schedule snapshot (gitignored under data/raw/ — lets the home page work offline). Kept
# separate from core.live's results cache: same feed, different projection of it.
FIXTURES_CACHE_PATH = config.RAW_DIR / "wc2026_fixtures.json"

# How long after kick-off a match with no full-time score is still treated as "live". 90' + a
# 15' break + stoppage ≈ 110'; knockouts can run to extra time + penalties. 130' is a deliberately
# generous, honest upper bound — a finished match flips to "finished" the moment the feed posts a
# score regardless, so this only governs the brief window before that.
LIVE_WINDOW = timedelta(minutes=130)

# Group-stage letters (must match the keys in data/wc2026.json).
_GROUP_LETTERS = set("ABCDEFGHIJKL")

# "13:00 UTC-6" / "20:00 UTC+5:30" / "18:00 UTC" / "18:00" — the trailing zone and its offset are
# both optional (bare "UTC" or no zone at all means UTC+0). Captures HH, MM, ±H, [MM].
_TIME_RE = re.compile(r"^\s*(\d{1,2}):(\d{2})(?:\s*UTC\s*([+-]?\d{1,2})?(?::?(\d{2}))?)?\s*$")


# --------------------------------------------------------------------------------------
# Pure parsing (no network — unit-tested directly)
# --------------------------------------------------------------------------------------
def _team_name(team) -> str:
    """Pull a team name out of an openfootball ``team1``/``team2`` value (string or {name})."""
    if isinstance(team, dict):
        return str(team.get("name", "")).strip()
    return str(team or "").strip()


def _group_letter(group) -> str | None:
    """Map an openfootball ``group`` field (e.g. ``"Group A"``) to its single-letter key.

    Returns ``None`` for knockout matches (no group) or anything that does not resolve to a
    known A–L group, so callers can cleanly treat those as neutral-venue ties.
    """
    if not group:
        return None
    token = str(group).strip().split()[-1].upper()
    return token if token in _GROUP_LETTERS else None


def parse_kickoff(date_str: str | None, time_str: str | None) -> datetime | None:
    """Resolve an openfootball ``date`` + ``time`` into an absolute, timezone-aware UTC instant.

    ``date_str`` is ``"YYYY-MM-DD"``; ``time_str`` is ``"HH:MM"`` optionally followed by a
    ``UTC±H[:MM]`` offset (the feed's local-to-venue stamp). With no offset the time is assumed to
    already be UTC. Returns ``None`` when either field is missing or unparseable — the caller then
    treats the fixture as merely "scheduled" (date known, exact kickoff unknown).
    """
    if not date_str:
        return None
    try:
        d = datetime.strptime(str(date_str).strip(), "%Y-%m-%d")
    except (ValueError, TypeError):
        return None
    if not time_str:
        return None
    m = _TIME_RE.match(str(time_str))
    if not m:
        return None
    hh, mm, off_h, off_m = m.group(1), m.group(2), m.group(3), m.group(4)
    if off_h is None:
        tz: timezone = UTC
    else:
        signed_h = int(off_h)  # int("+6")==6, int("-6")==-6
        minutes = int(off_m) if off_m else 0
        sign = -1 if signed_h < 0 or off_h.strip().startswith("-") else 1
        tz = timezone(timedelta(hours=signed_h, minutes=sign * minutes))
    try:
        local = datetime(d.year, d.month, d.day, int(hh), int(mm), tzinfo=tz)
    except ValueError:
        return None
    return local.astimezone(UTC)


def parse_fixtures(data: dict) -> list[dict]:
    """Extract every match in an openfootball ``worldcup.json`` payload into a normalized schema.

    Each fixture is ``{team1, team2, group, round, venue, date, kickoff_utc, finished,
    team1_score, team2_score, scorers1, scorers2}`` where team names are run through
    :func:`core.config.normalize_team` (so they join the committed draw cleanly), ``group`` is a
    single A–L letter or ``None`` (knockouts), and ``kickoff_utc`` is an ISO-8601 UTC string or
    ``None``. ``team1_score``/``team2_score`` align with ``team1``/``team2`` and are ``None`` until
    the match is played. Unlike :func:`core.live.parse_openfootball_matches`, *all* matches are
    kept — upcoming and knockout fixtures included — because the home page needs the full calendar.
    """
    out: list[dict] = []
    for m in data.get("matches", []):
        team1 = config.normalize_team(_team_name(m.get("team1")))
        team2 = config.normalize_team(_team_name(m.get("team2")))
        if not team1 or not team2:
            continue
        ft = (m.get("score") or {}).get("ft")
        played = (
            isinstance(ft, (list, tuple))
            and len(ft) == 2
            and ft[0] is not None
            and ft[1] is not None
        )
        kickoff = parse_kickoff(m.get("date"), m.get("time"))
        out.append(
            {
                "team1": team1,
                "team2": team2,
                "group": _group_letter(m.get("group")),
                "round": (str(m.get("round")).strip() or None) if m.get("round") else None,
                "venue": (str(m.get("ground")).strip() or None) if m.get("ground") else None,
                "date": (str(m.get("date")).strip() or None) if m.get("date") else None,
                "kickoff_utc": kickoff.isoformat() if kickoff else None,
                "finished": bool(played),
                "team1_score": int(ft[0]) if played else None,
                "team2_score": int(ft[1]) if played else None,
                "scorers1": _scorers(m.get("goals1")),
                "scorers2": _scorers(m.get("goals2")),
            }
        )
    return out


def _scorers(goals) -> list[str]:
    """Compact ``["Name 67'", ...]`` list from an openfootball ``goals1``/``goals2`` array."""
    if not isinstance(goals, (list, tuple)):
        return []
    out = []
    for g in goals:
        if not isinstance(g, dict):
            continue
        name = str(g.get("name", "")).strip()
        minute = str(g.get("minute", "")).strip()
        if name:
            out.append(f"{name} {minute}'" if minute else name)
    return out


# --------------------------------------------------------------------------------------
# Status + selection (take an explicit ``now`` so they stay pure and testable)
# --------------------------------------------------------------------------------------
def _kickoff_dt(fx: dict) -> datetime | None:
    iso = fx.get("kickoff_utc")
    if not iso:
        return None
    try:
        dt = datetime.fromisoformat(iso)
    except (ValueError, TypeError):
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def classify_status(fx: dict, now: datetime) -> str:
    """Classify a fixture against ``now`` → one of ``finished``/``live``/``upcoming``/
    ``recent``/``scheduled``.

    * ``finished`` — the feed has a full-time score.
    * ``live`` — kicked off, no score yet, still inside :data:`LIVE_WINDOW` (inferred — see module).
    * ``upcoming`` — kicks off in the future (exact time known).
    * ``recent`` — kicked off and past the live window but still no score in the feed (a result lag).
    * ``scheduled`` — date known but no parseable kickoff time, so it can't be ordered against now.
    """
    if fx.get("finished"):
        return "finished"
    ko = _kickoff_dt(fx)
    if ko is None:
        return "scheduled"
    if now < ko:
        return "upcoming"
    if now <= ko + LIVE_WINDOW:
        return "live"
    return "recent"


def current_match(fixtures: list[dict], now: datetime) -> dict | None:
    """The match to feature as live, or ``None``. If several overlap, the earliest kickoff wins."""
    live = [fx for fx in fixtures if classify_status(fx, now) == "live"]
    if not live:
        return None
    return min(live, key=lambda fx: _kickoff_dt(fx) or now)


def next_match(fixtures: list[dict], now: datetime) -> dict | None:
    """The soonest upcoming match (exact kickoff in the future), or ``None`` if none remain."""
    upcoming = [fx for fx in fixtures if classify_status(fx, now) == "upcoming" and _kickoff_dt(fx)]
    if not upcoming:
        return None
    return min(upcoming, key=lambda fx: _kickoff_dt(fx))


def has_remaining_matches(fixtures: list[dict], now: datetime) -> bool:
    """True while any match is still live or upcoming (used to detect a finished tournament).

    Only *timed* fixtures count — a date-only ``scheduled`` row has no kickoff to compare against
    ``now``, so it can't keep a long-finished tournament looking "live".
    """
    return any(classify_status(fx, now) in ("live", "upcoming") for fx in fixtures)


def todays_matches(fixtures: list[dict], now: datetime, tz) -> list[dict]:
    """All fixtures whose kickoff falls on *today* in the user's timezone, sorted by kickoff.

    Matches with a known kickoff are ordered by time; date-only fixtures (no parseable kickoff)
    whose calendar date equals today are appended afterwards. ``tz`` is a ``tzinfo`` (e.g. a
    ``zoneinfo.ZoneInfo``); the caller resolves it from the browser, defaulting to UTC.
    """
    today = now.astimezone(tz).date()
    timed, dateless = [], []
    for fx in fixtures:
        ko = _kickoff_dt(fx)
        if ko is not None:
            if ko.astimezone(tz).date() == today:
                timed.append(fx)
        elif fx.get("date"):
            try:
                if datetime.strptime(fx["date"], "%Y-%m-%d").date() == today:
                    dateless.append(fx)
            except (ValueError, TypeError):
                continue
    timed.sort(key=lambda fx: _kickoff_dt(fx))
    return timed + dateless


def kickoff_datetime(fx: dict) -> datetime | None:
    """Public accessor for a fixture's timezone-aware UTC kickoff (or ``None``)."""
    return _kickoff_dt(fx)


# --------------------------------------------------------------------------------------
# Snapshot cache (gitignored; lets the home page work offline from the last good fetch)
# --------------------------------------------------------------------------------------
def _empty_snapshot(error: str | None = None) -> dict:
    snap: dict = {"fetched_at": None, "source": SOURCE, "url": None, "fixtures": []}
    if error is not None:
        snap["error"] = error
    return snap


def _write_cache(snapshot: dict, cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")


def read_cache(cache_path: Path | None = None) -> dict | None:
    """Return the last cached schedule snapshot, or ``None`` if there is no (readable) cache."""
    cache_path = Path(cache_path) if cache_path is not None else FIXTURES_CACHE_PATH
    if not cache_path.exists():
        return None
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# --------------------------------------------------------------------------------------
# Fetch (I/O — injectable downloader keeps tests hermetic)
# --------------------------------------------------------------------------------------
def _http_get_json(url: str, timeout: float) -> dict:
    """Default downloader: GET a JSON document over HTTPS (trusted public mirror)."""
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 (trusted https mirror)
        return json.loads(resp.read().decode("utf-8"))


def fetch_fixtures(
    url: str | None = None,
    *,
    downloader=None,
    cache_path: Path | None = None,
    timeout: float = 30.0,
    allow_cache: bool = True,
) -> dict:
    """Fetch the full 2026 schedule and return a snapshot dict.

    The snapshot is ``{"fetched_at": <ISO utc | None>, "source": str, "url": str | None,
    "fixtures": [...]}``. On a successful download it is also written to ``cache_path``.

    Graceful degradation (so the home page always renders): if the download or parse fails, the
    last cached snapshot is returned when ``allow_cache`` (its ``source`` marked ``"… (cached)"``);
    if there is no cache, an empty snapshot with an ``error`` key is returned. ``downloader`` is
    injectable — ``(url, timeout) -> dict`` — so tests need no network.
    """
    url = url or config.WC2026_LIVE_URL
    cache_path = Path(cache_path) if cache_path is not None else FIXTURES_CACHE_PATH
    downloader = downloader or _http_get_json

    try:
        data = downloader(url, timeout)
        snapshot = {
            "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "source": SOURCE,
            "url": url,
            "fixtures": parse_fixtures(data),
        }
        _write_cache(snapshot, cache_path)
        return snapshot
    except Exception as exc:  # network/parse failure → degrade gracefully, never crash the app
        if allow_cache:
            cached = read_cache(cache_path)
            if cached is not None:
                cached = dict(cached)
                cached["source"] = f"{cached.get('source', SOURCE)} (cached)"
                return cached
        return _empty_snapshot(error=str(exc))


def build_schedule(snapshot_or_fixtures) -> list[dict]:
    """Project a fetched snapshot (or fixtures list) down to a *structure-only* committed schedule.

    Strips the volatile result fields (``finished``, scores, scorers), keeping only what is fixed
    once the draw is made — teams (or knockout placeholders), group/round, venue, date and the
    resolved UTC kick-off. This is what ``scripts/update_results.py --fixtures`` bakes into the
    committed ``data/wc2026.json`` ``fixtures[]`` block; live scores stay an optional overlay.
    """
    fixtures = (
        snapshot_or_fixtures.get("fixtures", [])
        if isinstance(snapshot_or_fixtures, dict)
        else snapshot_or_fixtures
    )
    return [
        {
            "team1": f["team1"],
            "team2": f["team2"],
            "group": f.get("group"),
            "round": f.get("round"),
            "venue": f.get("venue"),
            "date": f.get("date"),
            "kickoff_utc": f.get("kickoff_utc"),
        }
        for f in fixtures
    ]


# --------------------------------------------------------------------------------------
# Timezone resolution (pure; the UI passes the browser zone in, this never raises)
# --------------------------------------------------------------------------------------
def resolve_timezone(name: str | None) -> tuple[tzinfo, str]:
    """Resolve an IANA tz name to ``(tzinfo, label)``, falling back to Europe/Luxembourg, then UTC.

    Framework-free so it stays in ``core``: the Streamlit layer reads the browser zone
    (``st.context.timezone``) and passes the string here. Any unknown/blank/unresolvable name
    degrades to :data:`DEFAULT_TIMEZONE`, and only if the tz database itself is unavailable does it
    fall back to UTC — so the home page always renders sensible local kick-off times.
    """
    for candidate in (name, DEFAULT_TIMEZONE):
        if not candidate:
            continue
        try:
            return ZoneInfo(candidate), candidate
        except (ZoneInfoNotFoundError, ValueError):
            continue
    return UTC, "UTC"


def _coerce_tz(timezone: tzinfo | str | None) -> tuple[tzinfo, str]:
    """Accept a tzinfo, an IANA name, or ``None`` and return ``(tzinfo, label)``."""
    if timezone is None:
        return resolve_timezone(None)
    if isinstance(timezone, str):
        return resolve_timezone(timezone)
    return timezone, getattr(timezone, "key", None) or str(timezone)


# --------------------------------------------------------------------------------------
# Coarse, UI-facing status + "recently finished"
# --------------------------------------------------------------------------------------
def match_status(fixture: dict, now: datetime) -> str:
    """The UI-facing status: one of ``"upcoming"``, ``"live"``, ``"finished"``, ``"unknown"``.

    A coarsening of :func:`classify_status` (which keeps the finer ``"recent"`` / ``"scheduled"``
    engine states): a match the UI can't confidently place on the timeline — kicked off long ago
    with no score in the feed yet (result lag), or with no parseable kick-off time at all — is
    reported as ``"unknown"`` rather than guessed.
    """
    status = classify_status(fixture, now)
    return status if status in ("upcoming", "live", "finished") else "unknown"


def recently_finished(
    fixtures: list[dict],
    now: datetime,
    *,
    within: timedelta = timedelta(hours=48),
    limit: int | None = None,
) -> list[dict]:
    """Finished matches whose kick-off falls within the last ``within`` of ``now``, newest first.

    Ordering/recency keys off kick-off (the feed has no full-time timestamp); ``within`` is sized so
    a just-played match comfortably clears its own ~2-hour runtime. ``limit`` caps the list for a
    compact "Recently finished" strip. Fixtures without a parseable kick-off can't be ordered and
    are skipped.
    """
    out = []
    for fx in fixtures:
        if not fx.get("finished"):
            continue
        ko = _kickoff_dt(fx)
        if ko is None:
            continue
        if timedelta(0) <= now - ko <= within:
            out.append(fx)
    out.sort(key=lambda fx: _kickoff_dt(fx), reverse=True)
    return out[:limit] if limit else out


def format_time_until(delta: timedelta | float | int) -> str:
    """Humanize a positive time-to-kick-off, e.g. ``"3d 4h"``, ``"2h 15m"``, ``"7m"``, ``"<1m"``.

    Accepts a :class:`~datetime.timedelta` or a number of seconds. Non-positive input → ``"now"``.
    Drives the "Next match in Xh Ym" line (the live ticking clock is the iframe's client-side JS).
    """
    secs = int(delta.total_seconds() if isinstance(delta, timedelta) else delta)
    if secs <= 0:
        return "now"
    days, rem = divmod(secs, 86_400)
    hours, rem = divmod(rem, 3_600)
    minutes = rem // 60
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m"
    return "<1m"


# --------------------------------------------------------------------------------------
# UI-safe resolver: committed schedule (data/wc2026.json) + optional live-results overlay
# --------------------------------------------------------------------------------------
def _load_wc_json(path: Path | None = None) -> dict:
    """Read the committed tournament file as raw JSON, returning ``{}`` if unreadable (never raises)."""
    path = Path(path) if path is not None else config.WC2026_PATH
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _read_cache_snapshot(path: Path | None = None) -> dict | None:
    """Read the gitignored live-results snapshot (``data/raw/wc2026_live.json``), or ``None``."""
    path = Path(path) if path is not None else config.WC2026_LIVE_CACHE_PATH
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def load_committed_fixtures(wc: dict | None = None, *, path: Path | None = None) -> list[dict]:
    """Load the committed 2026 schedule (``data/wc2026.json`` ``fixtures[]``) as the offline base.

    Returns fixtures in the same schema :func:`parse_fixtures` produces, but result fields are
    defaulted to "not played" (``finished=False``, scores ``None``, empty scorers) — the committed
    block is structure only, so :func:`merge_results_into_schedule` stamps the live state on top.
    Team names are run through :func:`core.config.normalize_team` so committed, live and Elo
    spellings all join. UI-safe: no network, and a missing/empty block simply yields ``[]``.
    """
    if wc is None:
        wc = _load_wc_json(path)
    out: list[dict] = []
    for f in wc.get("fixtures") or []:
        team1 = config.normalize_team(_team_name(f.get("team1")))
        team2 = config.normalize_team(_team_name(f.get("team2")))
        if not team1 or not team2:
            continue
        out.append(
            {
                "team1": team1,
                "team2": team2,
                "group": f.get("group"),
                "round": f.get("round"),
                "venue": f.get("venue"),
                "date": f.get("date"),
                "kickoff_utc": f.get("kickoff_utc"),
                "finished": False,
                "team1_score": None,
                "team2_score": None,
                "scorers1": [],
                "scorers2": [],
            }
        )
    return out


def merge_results_into_schedule(schedule: list[dict], results: list[dict]) -> list[dict]:
    """Overlay played results onto the committed schedule, returning fresh fixture dicts.

    ``results`` is the simulator's ``known_results`` schema (``{home, away, home_score,
    away_score}``). Each is matched to its fixture by *unordered* team pair, oriented to that
    fixture's ``team1``/``team2``, and stamped ``finished`` with the score. Names are normalized so
    committed and live spellings join; an entry with no matching fixture (e.g. a knockout result, or
    a still-null score) is ignored. The input ``schedule`` is not mutated.
    """
    by_pair: dict[frozenset, tuple[str, object, object]] = {}
    for r in results or []:
        home = config.normalize_team(r.get("home"))
        away = config.normalize_team(r.get("away"))
        if home and away:
            by_pair[frozenset((home, away))] = (home, r.get("home_score"), r.get("away_score"))

    merged: list[dict] = []
    for fx in schedule:
        fx = dict(fx)
        hit = by_pair.get(frozenset((fx["team1"], fx["team2"])))
        if hit is not None and hit[1] is not None and hit[2] is not None:
            home, home_score, away_score = hit
            if home == fx["team1"]:
                fx["team1_score"], fx["team2_score"] = int(home_score), int(away_score)
            else:
                fx["team1_score"], fx["team2_score"] = int(away_score), int(home_score)
            fx["finished"] = True
        merged.append(fx)
    return merged


def _overlay_results(
    wc: dict, *, include_live_cache: bool, live_cache_path: Path | None
) -> list[dict]:
    """Collect the results overlay: committed ``known_results`` overridden by the live cache."""
    by_pair: dict[frozenset, dict] = {}
    for r in wc.get("known_results") or []:
        home, away = config.normalize_team(r.get("home")), config.normalize_team(r.get("away"))
        if home and away:
            by_pair[frozenset((home, away))] = r
    if include_live_cache:
        cache = _read_cache_snapshot(live_cache_path)
        for r in (cache or {}).get("known_results") or []:
            home, away = config.normalize_team(r.get("home")), config.normalize_team(r.get("away"))
            if home and away:
                by_pair[frozenset((home, away))] = r  # cached feed is fresher → overrides committed
    return list(by_pair.values())


def resolve_fixtures(
    *,
    wc: dict | None = None,
    live_results: list[dict] | None = None,
    schedule_path: Path | None = None,
    live_cache_path: Path | None = None,
    include_live_cache: bool = True,
) -> list[dict]:
    """The UI-safe fixture resolver: committed schedule + optional live-results overlay. No network.

    Loads the committed ``data/wc2026.json`` schedule and stamps played scores on top. The overlay's
    precedence is: an explicit ``live_results`` list (e.g. a fresh fetch the app already holds) wins;
    otherwise the committed ``known_results`` are used, overridden by the gitignored live-results
    cache when present and ``include_live_cache``. Everything is offline and never raises, so the
    Matchday page can always decide what to show — fresh feed or not.
    """
    if wc is None:
        wc = _load_wc_json(schedule_path)
    schedule = load_committed_fixtures(wc, path=schedule_path)
    if live_results is None:
        live_results = _overlay_results(
            wc, include_live_cache=include_live_cache, live_cache_path=live_cache_path
        )
    return merge_results_into_schedule(schedule, live_results)


def resolve_snapshot(*, include_live_cache: bool = True) -> dict:
    """Wrap :func:`resolve_fixtures` in the ``{fixtures, fetched_at, source}`` snapshot the home
    page renders. Always succeeds — the committed schedule is always present — so unlike
    :func:`fetch_fixtures` there is no ``error`` path and no network call on page load.
    """
    wc = _load_wc_json()
    fixtures = resolve_fixtures(wc=wc, include_live_cache=include_live_cache)
    cache = _read_cache_snapshot() if include_live_cache else None
    fetched_at = (cache or {}).get("fetched_at") or wc.get("known_results_as_of")
    source = "data/wc2026.json (committed schedule)"
    if (cache or {}).get("known_results"):
        source += " + live results (cached)"
    elif wc.get("known_results"):
        source += " + committed results"
    return {"fixtures": fixtures, "fetched_at": fetched_at, "source": source}


# --------------------------------------------------------------------------------------
# Public matchday API — (now, timezone) in, render-ready objects out
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class MatchdayContext:
    """Everything the Matchday landing page needs for *this moment*, in one render-ready object.

    The UI layer only reads these fields — all selection/timezone/status logic happens here so the
    page stays a pure view. ``headline`` says which section should lead: ``"live"`` (a match is on),
    ``"next"`` (count down to the next kick-off), ``"over"`` (tournament finished), or ``"idle"``
    (no schedule / nothing to show).
    """

    now: datetime
    timezone: tzinfo
    tz_label: str
    live: list[dict]
    next_match: dict | None
    next_kickoff: datetime | None
    next_in: str | None
    seconds_to_next: float | None
    today: list[dict]
    recently_finished: list[dict]
    tournament_over: bool
    headline: str


def get_live_matches(
    now: datetime, timezone: tzinfo | str | None = None, *, fixtures: list[dict] | None = None
) -> list[dict]:
    """All matches currently in progress (status ``"live"``), earliest kick-off first.

    ``timezone`` is accepted for a uniform signature but does not affect the result (live/upcoming
    are absolute-time comparisons). ``fixtures`` defaults to the committed-schedule resolver.
    """
    fixtures = resolve_fixtures() if fixtures is None else fixtures
    live = [fx for fx in fixtures if match_status(fx, now) == "live"]
    live.sort(key=lambda fx: _kickoff_dt(fx) or now)
    return live


def get_next_match(
    now: datetime, timezone: tzinfo | str | None = None, *, fixtures: list[dict] | None = None
) -> dict | None:
    """The soonest upcoming match (exact kick-off in the future), or ``None`` if none remain."""
    fixtures = resolve_fixtures() if fixtures is None else fixtures
    return next_match(fixtures, now)


def get_today_matches(
    now: datetime, timezone: tzinfo | str | None = None, *, fixtures: list[dict] | None = None
) -> list[dict]:
    """Every fixture kicking off on *today's* date in the user's ``timezone``, sorted by kick-off."""
    tz, _ = _coerce_tz(timezone)
    fixtures = resolve_fixtures() if fixtures is None else fixtures
    return todays_matches(fixtures, now, tz)


def get_matchday_context(
    now: datetime,
    timezone: tzinfo | str | None = None,
    *,
    fixtures: list[dict] | None = None,
    recent_within: timedelta = timedelta(hours=48),
    recent_limit: int | None = 6,
) -> MatchdayContext:
    """Resolve the full matchday picture for ``now`` — the one call the UI needs.

    Bundles the live matches, the next kick-off (with a humanized "in Xh Ym"), today's fixtures and
    the recently-finished list into a :class:`MatchdayContext`, and picks the ``headline`` section.
    ``fixtures`` defaults to the committed-schedule resolver, so this works offline and never raises.
    """
    tz, label = _coerce_tz(timezone)
    fixtures = resolve_fixtures() if fixtures is None else fixtures

    live = get_live_matches(now, tz, fixtures=fixtures)
    nxt = get_next_match(now, tz, fixtures=fixtures)
    ko = _kickoff_dt(nxt) if nxt else None
    seconds = (ko - now).total_seconds() if ko else None
    today = get_today_matches(now, tz, fixtures=fixtures)
    recent = recently_finished(fixtures, now, within=recent_within, limit=recent_limit)
    over = not has_remaining_matches(fixtures, now)

    if live:
        headline = "live"
    elif nxt:
        headline = "next"
    elif over and any(fx.get("finished") for fx in fixtures):
        headline = "over"
    else:
        headline = "idle"

    return MatchdayContext(
        now=now,
        timezone=tz,
        tz_label=label,
        live=live,
        next_match=nxt,
        next_kickoff=ko,
        next_in=format_time_until(seconds) if seconds is not None else None,
        seconds_to_next=seconds,
        today=today,
        recently_finished=recent,
        tournament_over=over,
        headline=headline,
    )
