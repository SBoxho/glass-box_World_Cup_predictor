"""Live 2026 World Cup results — fetch played group matches and lock them into the simulator.

The tournament simulator (:mod:`core.simulate`) already honors a ``known_results`` block: any
locked group match is replayed from its real score instead of being sampled. This module fills
that block from a free, no-key, public-domain feed so the simulator can run *forward* from the
current standings.

Design notes (kept deliberately strict so the rest of the project's guarantees hold):

* **Framework-free.** Like everything in ``core``, this module imports no Streamlit/FastAPI — the
  app/api layers call it. The Streamlit layer adds the caching/TTL and the refresh button.
* **Pure parse split from I/O.** :func:`parse_openfootball_matches` is a pure function (unit-tested
  with no network); :func:`fetch_live_results` does the download and accepts an injectable
  ``downloader`` so tests stay hermetic.
* **Cached + optional.** Each successful fetch is written to a gitignored cache snapshot. If the
  network is unavailable, the last cached snapshot is returned; if there is none, an empty snapshot
  is returned. The app therefore always works offline (it just won't have fresh results).
* **Group stage only.** Only played *group* matches are locked. Knockout results are intentionally
  not locked: the simulator re-draws the entire bracket from group standings, so a fixed knockout
  result has no slot to live in yet (a possible future enhancement).

Source: ``openfootball/worldcup.json`` (CC0 / public domain, no API key). See ``DATA_SOURCES.md``.
"""

from __future__ import annotations

import json
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from core import config

SOURCE = "openfootball/worldcup.json"


# --------------------------------------------------------------------------------------
# Pure parsing (no network — unit-tested directly)
# --------------------------------------------------------------------------------------
def _team_name(team) -> str:
    """Pull a team name out of an openfootball ``team1``/``team2`` value (string or {name})."""
    if isinstance(team, dict):
        return str(team.get("name", "")).strip()
    return str(team or "").strip()


def parse_openfootball_matches(data: dict) -> list[dict]:
    """Extract played *group-stage* results from an openfootball ``worldcup.json`` payload.

    Returns a list of ``{home, away, home_score, away_score}`` dicts in the exact schema the
    simulator's ``known_results`` block expects, with team names run through
    :func:`core.config.normalize_team` so they join cleanly to the committed group names.

    A match is included only when it is (a) group stage — a truthy ``group`` field — and
    (b) actually played — ``score.ft`` is a two-element list of non-null goals. Everything else
    (unplayed fixtures, knockout matches) is skipped.
    """
    out: list[dict] = []
    for m in data.get("matches", []):
        if not m.get("group"):  # group stage only; knockouts are not locked (sim re-draws them)
            continue
        ft = (m.get("score") or {}).get("ft")
        if not isinstance(ft, (list, tuple)) or len(ft) != 2:
            continue  # not played yet
        if ft[0] is None or ft[1] is None:
            continue
        home = config.normalize_team(_team_name(m.get("team1")))
        away = config.normalize_team(_team_name(m.get("team2")))
        if not home or not away:
            continue
        out.append(
            {
                "home": home,
                "away": away,
                "home_score": int(ft[0]),
                "away_score": int(ft[1]),
            }
        )
    return out


# --------------------------------------------------------------------------------------
# Snapshot cache (gitignored; lets the app work offline from the last good fetch)
# --------------------------------------------------------------------------------------
def _empty_snapshot(error: str | None = None) -> dict:
    snap: dict = {"fetched_at": None, "source": SOURCE, "url": None, "known_results": []}
    if error is not None:
        snap["error"] = error
    return snap


def _write_cache(snapshot: dict, cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")


def read_cache(cache_path: Path | None = None) -> dict | None:
    """Return the last cached snapshot, or ``None`` if there is no (readable) cache."""
    cache_path = Path(cache_path) if cache_path is not None else config.WC2026_LIVE_CACHE_PATH
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


def fetch_live_results(
    url: str | None = None,
    *,
    downloader=None,
    cache_path: Path | None = None,
    timeout: float = 30.0,
    allow_cache: bool = True,
) -> dict:
    """Fetch played group results and return a snapshot dict.

    The snapshot is ``{"fetched_at": <ISO utc | None>, "source": str, "url": str | None,
    "known_results": [...]}``. On a successful download it is also written to ``cache_path``.

    Graceful degradation (so the app always works offline): if the download or parse fails,
    the last cached snapshot is returned when ``allow_cache`` (its ``source`` is marked
    ``"… (cached)"``); if there is no cache, an empty snapshot is returned. ``downloader`` is
    injectable — ``(url, timeout) -> dict`` — so tests need no network.
    """
    url = url or config.WC2026_LIVE_URL
    cache_path = Path(cache_path) if cache_path is not None else config.WC2026_LIVE_CACHE_PATH
    downloader = downloader or _http_get_json

    try:
        data = downloader(url, timeout)
        snapshot = {
            "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "source": SOURCE,
            "url": url,
            "known_results": parse_openfootball_matches(data),
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


# --------------------------------------------------------------------------------------
# Merge into a tournament definition
# --------------------------------------------------------------------------------------
def merge_known_results(wc: dict, results: list[dict]) -> dict:
    """Return a shallow copy of ``wc`` with ``results`` merged into its ``known_results``.

    Keyed by the unordered team pair, so each fixture appears once and a live result overrides any
    committed one. Names are assumed already normalized (both the committed block and the parsed
    results pass through :func:`core.config.normalize_team`).
    """
    by_pair: dict[frozenset, dict] = {
        frozenset((r["home"], r["away"])): r for r in wc.get("known_results", [])
    }
    for r in results:
        by_pair[frozenset((r["home"], r["away"]))] = r
    merged = dict(wc)
    merged["known_results"] = list(by_pair.values())
    return merged
