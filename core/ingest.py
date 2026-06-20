"""Data layer: download and clean the international match history, and load the committed
2026 tournament structure.

The cleaned match frame is the single input to everything downstream (Elo, features,
training). Cleaning is deterministic and side-effect-free apart from the cached download.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pandas as pd

from core import config, knockout

# Columns we rely on from the upstream results.csv schema.
_RAW_COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "city",
    "country",
    "neutral",
]


def download_results(force: bool = False, url: str | None = None) -> Path:
    """Download results.csv to the raw-data cache, returning its path.

    Cached: re-downloads only when the file is missing or ``force=True``. No API key needed
    — this is a public GitHub mirror of the widely used martj42/international_results set.
    """
    dest = config.RAW_RESULTS_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        return dest
    src = url or config.RESULTS_URL
    with urllib.request.urlopen(src, timeout=60) as resp:  # noqa: S310 (trusted https mirror)
        dest.write_bytes(resp.read())
    return dest


def load_raw_results(path: Path | None = None) -> pd.DataFrame:
    """Read the cached results.csv into a DataFrame (no cleaning yet)."""
    path = path or config.RAW_RESULTS_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run download_results() (or scripts/build_dataset.py) first."
        )
    return pd.read_csv(path)


def clean_results(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw results.csv into a tidy, chronologically sorted match frame.

    Steps: normalize team names, parse dates, drop unplayed rows, derive the {H,D,A} label
    and goal margin, and sort ascending by date (critical: every point-in-time feature relies
    on chronological order). Full history is kept here; the *training* window is filtered later.
    """
    df = df.copy()

    # Team-name normalization so the dataset, the draw, and squads all join on one spelling.
    df["home_team"] = df["home_team"].map(config.normalize_team)
    df["away_team"] = df["away_team"].map(config.normalize_team)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Drop rows that cannot anchor a result (unplayed fixtures, parse failures).
    df = df.dropna(subset=["date", "home_score", "away_score", "home_team", "away_team"])

    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    # The upstream `neutral` column is a boolean-like; coerce robustly to real bools.
    df["neutral"] = df["neutral"].astype(str).str.strip().str.lower().isin({"true", "1", "yes"})

    df["tournament"] = df["tournament"].fillna("").astype(str)
    df["country"] = df.get("country", pd.Series(index=df.index, dtype=str)).fillna("").astype(str)

    df["margin"] = df["home_score"] - df["away_score"]
    df["result"] = "D"
    df.loc[df["margin"] > 0, "result"] = "H"
    df.loc[df["margin"] < 0, "result"] = "A"

    df = df.sort_values("date", kind="mergesort").reset_index(drop=True)
    df["year"] = df["date"].dt.year

    keep = [
        "date",
        "year",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "margin",
        "result",
        "tournament",
        "country",
        "neutral",
    ]
    return df[keep]


def get_clean_matches(force_download: bool = False) -> pd.DataFrame:
    """Convenience: download (if needed) + load + clean, returning the match frame."""
    download_results(force=force_download)
    return clean_results(load_raw_results())


# --------------------------------------------------------------------------------------
# 2026 tournament structure
# --------------------------------------------------------------------------------------
def load_wc2026(path: Path | None = None) -> dict:
    """Load and validate the committed 2026 tournament definition (groups, hosts, bracket)."""
    path = path or config.WC2026_PATH
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    # Normalize every team spelling on the way in so downstream joins are clean.
    data["groups"] = {
        g: [config.normalize_team(t) for t in teams] for g, teams in data["groups"].items()
    }
    data["hosts"] = [config.normalize_team(t) for t in data["hosts"]]
    data["host_groups"] = {config.normalize_team(t): g for t, g in data["host_groups"].items()}
    for kr in data.get("known_results", []):
        kr["home"] = config.normalize_team(kr["home"])
        kr["away"] = config.normalize_team(kr["away"])

    validate_wc2026(data)
    return data


def validate_wc2026(data: dict) -> None:
    """Assert the structural invariants of the 48-team / 12-group format.

    Raises AssertionError with a clear message on any violation. Cheap to call; the app and
    the simulation both rely on these invariants holding.
    """
    groups = data["groups"]
    assert len(groups) == 12, f"expected 12 groups, got {len(groups)}"
    assert set(groups) == set("ABCDEFGHIJKL"), f"groups must be A..L, got {sorted(groups)}"

    all_teams = [t for teams in groups.values() for t in teams]
    assert all(len(teams) == 4 for teams in groups.values()), (
        "every group must have exactly 4 teams"
    )
    assert len(all_teams) == 48, f"expected 48 teams, got {len(all_teams)}"
    assert len(set(all_teams)) == 48, "team names must be unique across all groups"

    for host, grp in data["host_groups"].items():
        assert grp in groups, f"host {host} assigned to unknown group {grp}"
        assert host in groups[grp], f"host {host} not present in its own group {grp}"

    assert data.get("best_thirds_count") == 8, "2026 format takes the 8 best third-placed teams"

    # The knockout bracket + Annexe C now come from the official rules file (the single source of
    # truth, loaded by core.knockout); assert its structural invariants here so a malformed rules
    # file is caught at load time rather than mid-simulation.
    spec = knockout.build_spec(knockout.load_rules())
    r32 = [m for m in spec.matches if m.round == "R32"]
    assert len(r32) == 16, f"Round of 32 needs exactly 16 matches, got {len(r32)}"
    ids = [m.id for m in spec.matches]
    assert ids == [f"M{n}" for n in range(73, 105)], "knockout matches must be M73..M104 in order"
    assert len(spec.annex_matrix) == 495, (
        f"Annexe C must have 495 third-place rows, got {len(spec.annex_matrix)}"
    )
    annex_slots = {m.b for m in r32 if m.b.startswith(knockout.ANNEX_PREFIX)}
    assert len(annex_slots) == 8, (
        f"expected 8 Annexe-eligible group-winner slots, got {len(annex_slots)}"
    )
