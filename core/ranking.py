"""FIFA men's world ranking — a time-indexed points table, attached point-in-time to matches.

The ranking is the second external signal (after Elo). It is loaded as a single tidy frame of
``(team, total_points, date)`` snapshots and queried *as of* a date — exactly the discipline
:func:`core.elo.ratings_as_of` uses for Elo — so a match dated ``D`` only ever sees a ranking
published on or before ``D`` (no leakage).

Two sources are concatenated into one time series:

* **History** — a community compilation of public FIFA ranking data (1992 → 2024), downloaded and
  cached like the match results (no API key). See ``DATA_SOURCES.md``.
* **Current snapshot** — a small committed table of the 48 WC-2026 teams' current points (public
  facts), appended as one more dated snapshot so 2026 predictions use up-to-date points rather than
  the stale tail of the history feed.

This module is framework-free (``core`` rule). Only ``total_points`` is used as a feature; FIFA's
points method changed in 2018, but the model feature is always a *same-date* difference between two
teams, so the within-match scale is consistent (the cross-era distribution shift is documented).
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pandas as pd

from core import config

_COLUMNS = ["team", "total_points", "date"]


# --------------------------------------------------------------------------------------
# Download + load
# --------------------------------------------------------------------------------------
def download_ranking_history(force: bool = False, url: str | None = None) -> Path:
    """Download the FIFA ranking history CSV to the raw-data cache, returning its path.

    Cached: re-downloads only when missing or ``force=True`` (mirrors
    :func:`core.ingest.download_results`). No API key required.
    """
    dest = config.FIFA_RANKING_CACHE_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        return dest
    src = url or config.FIFA_RANKING_URL
    with urllib.request.urlopen(src, timeout=120) as resp:  # noqa: S310 (trusted https mirror)
        dest.write_bytes(resp.read())
    return dest


def load_ranking_history(path: Path | None = None) -> pd.DataFrame:
    """Read the cached ranking history into a tidy ``(team, total_points, date)`` frame.

    Team names are normalized so they join to the match history and the 2026 draw; rows without a
    parseable date or points are dropped; sorted ascending by date.
    """
    path = path or config.FIFA_RANKING_CACHE_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run download_ranking_history() (or scripts/build_dataset.py) first."
        )
    return _tidy(pd.read_csv(path))


def load_current_snapshot(path: Path | None = None) -> pd.DataFrame:
    """Read the committed current-ranking snapshot into the same tidy frame.

    Schema: ``{"as_of": "YYYY-MM-DD", "ranking": [{"team", "points", "rank"}, ...]}``. Returns an
    empty (correctly-typed) frame if the file is absent, so the pipeline still runs on history alone.
    """
    path = path or config.FIFA_RANKING_2026_PATH
    if not path.exists():
        return _empty()
    data = json.loads(path.read_text(encoding="utf-8"))
    as_of = data["as_of"]
    rows = [
        {"team": r["team"], "total_points": r["points"], "date": as_of} for r in data["ranking"]
    ]
    return _tidy(pd.DataFrame(rows, columns=_COLUMNS))


def load_rankings(
    *, download: bool = True, force_download: bool = False, snapshot: bool = True
) -> pd.DataFrame:
    """Return the full ranking time series: history (+ current snapshot), tidy and date-sorted.

    The single frame everything downstream queries via :func:`points_as_of`. ``download`` controls
    fetching the history (cached); ``snapshot`` appends the committed 2026 snapshot.
    """
    if download:
        download_ranking_history(force=force_download)
    frames = [load_ranking_history()]
    if snapshot:
        snap = load_current_snapshot()
        if not snap.empty:
            frames.append(snap)
    out = pd.concat(frames, ignore_index=True)
    # If history and snapshot collide on (team, date), keep the snapshot (last) value.
    out = out.drop_duplicates(subset=["team", "date"], keep="last")
    return out.sort_values("date", kind="mergesort").reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Point-in-time query (mirrors core.elo.ratings_as_of)
# --------------------------------------------------------------------------------------
def points_as_of(
    rankings: pd.DataFrame, as_of_date: pd.Timestamp | None = None
) -> dict[str, float]:
    """Return each team's FIFA points as of a date: the latest snapshot with ``date <= as_of_date``.

    ``as_of_date=None`` gives current/latest points (the committed snapshot, since it is the most
    recent date). Uses ``<=`` (allow same-date) to match the ``merge_asof`` in
    :func:`core.features.build_features` exactly — both must agree for the no-leakage test to pass.
    """
    if rankings is None or rankings.empty:
        return {}
    df = rankings if as_of_date is None else rankings[rankings["date"] <= pd.Timestamp(as_of_date)]
    if df.empty:
        return {}
    latest = df.loc[df.groupby("team")["date"].idxmax()]
    return dict(zip(latest["team"], latest["total_points"].astype(float), strict=True))


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _tidy(df: pd.DataFrame) -> pd.DataFrame:
    out = df[_COLUMNS].copy()
    out["team"] = out["team"].map(config.normalize_team)
    out["total_points"] = pd.to_numeric(out["total_points"], errors="coerce")
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=_COLUMNS)
    return out.sort_values("date", kind="mergesort").reset_index(drop=True)


def _empty() -> pd.DataFrame:
    return pd.DataFrame(
        {"team": [], "total_points": pd.Series([], dtype=float), "date": []}
    ).astype({"team": "object", "total_points": "float64"})
