"""Squad strength — a time-indexed per-nation rating table, attached point-in-time to matches.

The third external signal (after Elo and the FIFA ranking). Each national team is summarised, for
every EA Sports FC / FIFA game version, by a handful of squad-strength *components* derived from the
players' overall ratings:

* ``bestxi_ovr`` — mean overall of the best XI (top 11 players by rating);
* ``attack_ovr`` / ``def_ovr`` — mean overall of the best attackers / defenders (line strength);
* ``depth_ovr`` — mean overall of players ranked 12–26 (bench depth);
* ``star3_ovr`` — mean overall of the top 3 players (star power).

These feed four leak-free model features (see :data:`core.config.SQUAD_FEATURES`), each a point-in-
time **home − away** difference attached via :func:`strength_as_of` — exactly the discipline
:func:`core.ranking.points_as_of` uses for FIFA points. A match dated ``D`` only ever sees a ratings
*version* released on or before ``D``.

Two sources are concatenated into one time series (mirroring :mod:`core.ranking`):

* **History** — versioned community ratings (FIFA 15 → FC 24), a sofifa-derived "legacy" complete-
  player dataset downloaded from a public Hugging Face mirror (no auth) and cached under
  ``data/external/`` (gitignored — proprietary dumps are never committed). See ``DATA_SOURCES.md``.
* **Current snapshot** — the committed :data:`core.config.SQUADS_2026_PATH` (the 48 WC-2026 squads,
  EA FC 26), appended as one more dated version so 2026 predictions use current squads.

This module is framework-free (``core`` rule). **In-game ratings are third-party estimates, not
official data** — surfaced as such in the UI and docs.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from core import config

# The per-(nation, date) strength components stored in the time series. ``squad_strength.parquet``
# caches exactly these columns; the synthetic test fixture mirrors them.
METRICS = ["bestxi_ovr", "attack_ovr", "def_ovr", "depth_ovr", "star3_ovr"]
COLUMNS = ["team", "date", *METRICS]

# Default components for a nation absent from a version — every metric at the fixed SQUAD_OVR_BASE,
# so two unrated sides give a 0 difference on every squad feature (the no-leakage identity).
DEFAULT_COMPONENTS: dict[str, float] = dict.fromkeys(METRICS, config.SQUAD_OVR_BASE)

# Primary-position buckets (first listed position, upper-cased) for the attack / defence lines.
_ATTACK_POS = {"ST", "CF", "LW", "RW", "LF", "RF", "LS", "RS"}
_DEFENCE_POS = {"CB", "LB", "RB", "LWB", "RWB", "LCB", "RCB", "SW", "GK"}
_LINE_N = 5  # players averaged per line
_LINE_MIN = 3  # below this in a bucket, fall back to the overall best-XI level


# --------------------------------------------------------------------------------------
# Diff formulas — the SINGLE definition shared by every feature path.
# --------------------------------------------------------------------------------------
def _diffs_from_components(home_c, away_c) -> dict:
    """The four squad model features from two component mappings (home, away).

    Works with plain ``float`` components (single-match / inference paths) *and* with pandas
    ``Series`` components (the vectorized batch path), since it only adds and subtracts — so the
    batch and single-match features are identical by construction. Each diff negates under a
    home/away swap (``attack_vs_def`` too — verified in the symmetry test), so
    :func:`core.model.reverse_features` just flips the sign.
    """
    return {
        "squad_strength_diff": home_c["bestxi_ovr"] - away_c["bestxi_ovr"],
        # Each side's attack-vs-the-other's-defence edge, differenced so it is antisymmetric:
        #   (home_att − away_def) − (away_att − home_def) = (home_att+home_def) − (away_att+away_def)
        "attack_vs_def": (home_c["attack_ovr"] - away_c["def_ovr"])
        - (away_c["attack_ovr"] - home_c["def_ovr"]),
        "depth_diff": home_c["depth_ovr"] - away_c["depth_ovr"],
        "star_power_diff": home_c["star3_ovr"] - away_c["star3_ovr"],
    }


# --------------------------------------------------------------------------------------
# Download + load (mirrors core.ranking)
# --------------------------------------------------------------------------------------
def download_squad_ratings(force: bool = False, url: str | None = None) -> Path:
    """Download the versioned player-ratings CSV to ``data/external/`` (gitignored), returning path.

    Cached: re-downloads only when missing or ``force=True`` (mirrors
    :func:`core.ranking.download_ranking_history`). No API key required — a public mirror.
    """
    dest = config.SQUAD_RATINGS_CACHE_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        return dest
    src = url or config.SQUAD_RATINGS_URL
    req = urllib.request.Request(src, headers={"User-Agent": "glass-box-wc/1.0"})
    with urllib.request.urlopen(req, timeout=300) as resp:  # noqa: S310 (trusted https mirror)
        dest.write_bytes(resp.read())
    return dest


def load_squad_history(path: Path | None = None) -> pd.DataFrame:
    """Build the per-(nation, version-release-date) strength table from the cached ratings CSV.

    For each FIFA version we take the launch-day roster (the earliest ``fifa_update_date`` in that
    version — a deterministic single point in time), aggregate every nation's players into the
    :data:`METRICS` components, and date the row at the version's release. Nations with fewer than
    ``SQUAD_MIN_PLAYERS`` rated players in a version are dropped (they fall back to the default at
    query time). Team names are normalized so they join to the match history and the 2026 draw.
    """
    path = path or config.SQUAD_RATINGS_CACHE_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run download_squad_ratings() (or scripts/build_dataset.py) first."
        )
    usecols = [
        "fifa_version",
        "fifa_update_date",
        "nationality_name",
        "player_positions",
        "overall",
    ]
    raw = pd.read_csv(path, usecols=usecols, low_memory=False)
    raw["fifa_update_date"] = pd.to_datetime(raw["fifa_update_date"], errors="coerce")
    raw["overall"] = pd.to_numeric(raw["overall"], errors="coerce")
    raw = raw.dropna(subset=["fifa_version", "fifa_update_date", "nationality_name", "overall"])

    # Keep only the launch-day roster of each version (one rating per player per version).
    release = raw.groupby("fifa_version")["fifa_update_date"].transform("min")
    launch = raw[raw["fifa_update_date"] == release]

    rows = []
    for (_, version_date), grp in launch.groupby(["fifa_version", "fifa_update_date"]):
        for nation, players in grp.groupby("nationality_name"):
            if len(players) < config.SQUAD_MIN_PLAYERS:
                continue
            comp = _components_from_players(
                players["overall"].to_numpy(), players["player_positions"].tolist()
            )
            rows.append({"team": config.normalize_team(nation), "date": version_date, **comp})
    return _tidy(pd.DataFrame(rows, columns=COLUMNS))


def load_current_snapshot(path: Path | None = None) -> pd.DataFrame:
    """Read the committed current-squads snapshot into the same per-(nation, date) component frame.

    Schema: ``{"as_of": "YYYY-MM-DD", "squads": {"<Nation>": [{"fc26_ovr", "position", ...}, ...]}}``.
    Returns an empty (correctly-typed) frame if the file is absent, so the pipeline still runs on
    history alone.
    """
    path = path or config.SQUADS_2026_PATH
    if not path.exists():
        return _empty()
    data = json.loads(path.read_text(encoding="utf-8"))
    as_of = pd.Timestamp(data["as_of"])
    rows = []
    for nation, players in data["squads"].items():
        if not players:
            continue
        overalls = np.array([p["fc26_ovr"] for p in players], dtype=float)
        positions = [p.get("position", "") for p in players]
        comp = _components_from_players(overalls, positions)
        rows.append({"team": config.normalize_team(nation), "date": as_of, **comp})
    return _tidy(pd.DataFrame(rows, columns=COLUMNS))


def load_squad_strength(
    *,
    download: bool = True,
    force_download: bool = False,
    history: bool = True,
    snapshot: bool = True,
) -> pd.DataFrame:
    """Return the full squad-strength time series: history (+ current snapshot), tidy, date-sorted.

    The single frame everything downstream queries via :func:`strength_as_of`. ``history`` reads the
    downloaded versioned ratings (needs the cache / ``download``); ``snapshot`` appends the committed
    2026 squads. For *inference only*, ``history=False`` gives a snapshot-only table — enough for
    current-strength predictions without the heavy historical download (used by the app loader).
    """
    frames = []
    if history:
        if download:
            download_squad_ratings(force=force_download)
        frames.append(load_squad_history())
    if snapshot:
        snap = load_current_snapshot()
        if not snap.empty:
            frames.append(snap)
    if not frames:
        return _empty()
    out = pd.concat(frames, ignore_index=True)
    # If history and snapshot collide on (team, date), keep the snapshot (last) value.
    out = out.drop_duplicates(subset=["team", "date"], keep="last")
    return out.sort_values("date", kind="mergesort").reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Point-in-time query (mirrors core.ranking.points_as_of)
# --------------------------------------------------------------------------------------
def strength_as_of(
    squads: pd.DataFrame, as_of_date: pd.Timestamp | None = None
) -> dict[str, dict[str, float]]:
    """Each team's squad components as of a date: the latest version with ``date <= as_of_date``.

    ``as_of_date=None`` gives current/latest components (the committed snapshot, the most recent
    date). Uses ``<=`` (allow same-date) to match the ``merge_asof`` in
    :func:`core.features.build_features` exactly — both must agree for the no-leakage test to pass.
    """
    if squads is None or squads.empty:
        return {}
    df = squads if as_of_date is None else squads[squads["date"] <= pd.Timestamp(as_of_date)]
    if df.empty:
        return {}
    latest = df.loc[df.groupby("team")["date"].idxmax()]
    return {row["team"]: {m: float(row[m]) for m in METRICS} for _, row in latest.iterrows()}


# --------------------------------------------------------------------------------------
# Current 26-man squads for the UI (committed snapshot, normalized team keys)
# --------------------------------------------------------------------------------------
def current_squads(path: Path | None = None) -> dict[str, list[dict]]:
    """Return ``{normalized nation: [player dict, ...]}`` from the committed snapshot (UI display).

    Players keep their raw fields (``name, club, position, fc26_ovr`` + the per-attribute fields)
    and are returned sorted by overall, descending. Empty dict if the snapshot is absent.
    """
    path = path or config.SQUADS_2026_PATH
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, list[dict]] = {}
    for nation, players in data.get("squads", {}).items():
        ordered = sorted(players, key=lambda p: p.get("fc26_ovr", 0), reverse=True)
        out[config.normalize_team(nation)] = ordered
    return out


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _components_from_players(overalls: np.ndarray, positions: list[str]) -> dict[str, float]:
    """Aggregate one nation's player overalls (+ primary positions) into the strength components.

    ``positions`` are raw ``player_positions`` strings (e.g. ``"ST, CF"``); the first listed
    position is the primary one used for line buckets. Line averages fall back to the best-XI level
    when a nation has too few players in a bucket.
    """
    ovr = np.sort(np.asarray(overalls, dtype=float))[::-1]
    bestxi = float(ovr[:11].mean())
    star3 = float(ovr[:3].mean())
    depth = float(ovr[11:26].mean()) if len(ovr) > 11 else bestxi

    primary = [str(p).split(",")[0].strip().upper() for p in positions]
    primary = np.array(primary)
    ovr_unsorted = np.asarray(overalls, dtype=float)

    def _line(posset: set[str]) -> float:
        mask = np.isin(primary, list(posset))
        sub = np.sort(ovr_unsorted[mask])[::-1]
        return float(sub[:_LINE_N].mean()) if len(sub) >= _LINE_MIN else bestxi

    return {
        "bestxi_ovr": bestxi,
        "attack_ovr": _line(_ATTACK_POS),
        "def_ovr": _line(_DEFENCE_POS),
        "depth_ovr": depth,
        "star3_ovr": star3,
    }


def _tidy(df: pd.DataFrame) -> pd.DataFrame:
    out = df[COLUMNS].copy()
    out["team"] = out["team"].map(config.normalize_team)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for m in METRICS:
        out[m] = pd.to_numeric(out[m], errors="coerce")
    out = out.dropna(subset=COLUMNS)
    return out.sort_values("date", kind="mergesort").reset_index(drop=True)


def _empty() -> pd.DataFrame:
    cols = {"team": pd.Series([], dtype="object"), "date": pd.Series([], dtype="datetime64[ns]")}
    cols.update({m: pd.Series([], dtype="float64") for m in METRICS})
    return pd.DataFrame(cols)
