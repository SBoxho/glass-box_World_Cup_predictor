"""Central configuration: paths, constants, the canonical feature list, and team-name
normalization. Everything else in ``core`` imports from here so there is a single source
of truth for things that must agree across modules (feature order, RNG seed, Elo tuning).

This module has **no** Streamlit / Flask / FastAPI imports — keeping ``core`` UI-agnostic
is the architectural rule of the repo (``app`` and ``api`` import from here, never vice versa).
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

WC2026_PATH = DATA_DIR / "wc2026.json"
# Official 2026 progression rules — the knockout match DAG (M73→M104), FIFA's 495-row Annexe C
# third-place matrix, and the group tiebreaker order. The single source of truth for the knockout
# structure and best-third slotting (see core.knockout); committed, machine-readable.
WC2026_RULES_PATH = DATA_DIR / "fifa_world_cup_2026_rules.json"
RAW_RESULTS_PATH = RAW_DIR / "results.csv"
WC2026_LIVE_CACHE_PATH = RAW_DIR / "wc2026_live.json"  # cached live-results snapshot (gitignored)
FIFA_RANKING_CACHE_PATH = (
    RAW_DIR / "fifa_ranking_history.csv"
)  # cached ranking history (gitignored)
FIFA_RANKING_2026_PATH = DATA_DIR / "fifa_ranking_2026.json"  # committed current-ranking snapshot
# Squad strength (Phase 3): raw versioned player ratings are downloaded to data/external/
# (gitignored — never commit proprietary dumps); only the small committed 2026 snapshot lives in
# data/. EXTERNAL_DIR is reserved for these third-party rating dumps.
EXTERNAL_DIR = DATA_DIR / "external"
SQUAD_RATINGS_CACHE_PATH = (
    EXTERNAL_DIR / "fifa_players_legacy.csv"
)  # downloaded history (gitignored)
SQUADS_2026_PATH = DATA_DIR / "squads2026.json"  # committed current-squads snapshot
MATCHES_PATH = PROCESSED_DIR / "matches.parquet"
FEATURES_PATH = PROCESSED_DIR / "features.parquet"
RANKINGS_PATH = PROCESSED_DIR / "rankings.parquet"
SQUAD_STRENGTH_PATH = PROCESSED_DIR / "squad_strength.parquet"  # derived per-(nation,version) cache
ELO_HISTORY_PATH = PROCESSED_DIR / "elo_history.parquet"

MODEL_PATH = MODELS_DIR / "model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
RELIABILITY_PLOT_PATH = MODELS_DIR / "reliability.png"
SHAP_SUMMARY_PATH = MODELS_DIR / "shap_summary.png"

# Source for international match history (no auth required — public GitHub mirror).
RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

# Live 2026 World Cup results (no auth required — public-domain/CC0 openfootball mirror). Used to
# lock already-played group matches so the tournament simulator runs forward from current standings.
WC2026_LIVE_URL = (
    "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2026/worldcup.json"
)

# Historical FIFA men's world ranking (no auth required — community compilation of public FIFA
# data). A single combined CSV (team, total_points, date) spanning 1992 → 2024; attached
# point-in-time to each match. Combined with the committed FIFA_RANKING_2026_PATH snapshot so
# current 2026 predictions use up-to-date points. See DATA_SOURCES.md.
FIFA_RANKING_URL = (
    "https://raw.githubusercontent.com/Dato-Futbol/fifa-ranking/master/ranking_fifa_historical.csv"
)

# Default FIFA points for a team with no ranking on/before a given date (e.g. unranked early-era
# sides). A fixed constant so the batch and single-match feature paths agree exactly — the
# no-leakage guardrail depends on that identity.
FIFA_POINTS_BASE = 1000.0

# Historical versioned EA Sports FC / FIFA player ratings (FIFA 15 → FIFA 23, ~one release/year),
# used to build a point-in-time per-nation squad-strength table. No auth required — a public Hugging
# Face mirror of the community sofifa-derived "legacy" complete-player dataset (one row per player
# per FIFA version, with nationality, overall, positions and the six attribute aggregates).
# Downloaded at build time and cached under data/external/ (gitignored). The committed
# SQUADS_2026_PATH snapshot is appended as the latest version so 2026 predictions use current
# squads. See DATA_SOURCES.md. In-game ratings are THIRD-PARTY ESTIMATES, not official data.
SQUAD_RATINGS_URL = (
    "https://huggingface.co/datasets/jsulz/FIFA23/resolve/main/male_players%20%28legacy%29.csv"
)

# Current EA FC 26 player ratings used to (re)generate the committed 2026 squads snapshot via
# scripts/build_squads_snapshot.py — same sofifa-derived legacy schema (fifa_version == 26). No auth
# required (public GitHub mirror). The raw CSV is cached under data/external/ (gitignored); only the
# small derived data/squads2026.json is committed.
SQUADS_2026_SOURCE_URL = (
    "https://raw.githubusercontent.com/ismailoksuz/EAFC26-DataHub/main/data/players.csv"
)
SQUADS_2026_RAW_CACHE_PATH = (
    EXTERNAL_DIR / "fc26_players.csv"
)  # downloaded FC26 ratings (gitignored)
SQUADS_PER_TEAM = 26  # players kept per nation in the committed snapshot (top-N by overall)

# Default squad overall for a nation absent from a ratings version (e.g. a minnow with too few
# rated players, or any team before FIFA 15). A fixed constant — like FIFA_POINTS_BASE — so the
# batch and single-match feature paths produce identical squad diffs (the no-leakage test needs
# that exact identity). Two unrated sides therefore give a 0 difference on every squad feature.
SQUAD_OVR_BASE = 60.0

# Minimum rated players a nation needs in a version to be aggregated (else it falls back to the
# default). Keeps a handful of stray rows from inventing a "squad" for a barely-represented nation.
SQUAD_MIN_PLAYERS = 16

# --------------------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------------------
SEED = 42

# --------------------------------------------------------------------------------------
# Outcome classes — fixed order used EVERYWHERE (model, SHAP, simulation, UI).
# H = home win, D = draw, A = away win. Swapping home/away maps H<->A and leaves D.
# --------------------------------------------------------------------------------------
CLASSES = ["H", "D", "A"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# --------------------------------------------------------------------------------------
# Feature engineering knobs
# --------------------------------------------------------------------------------------
FORM_WINDOW = 10  # matches looked back for form / goal-difference
H2H_HALFLIFE_YEARS = 8.0  # exponential decay half-life for head-to-head record
H2H_PRIOR = 0.5  # neutral score-share when two teams have never met
MAX_REST_DAYS = 60  # cap on days_rest so long gaps don't dominate

# Only matches on/after this year enter the *training* set (dense, modern-era signal).
# Full history is still used to warm up Elo and the rolling form windows.
TRAIN_START_YEAR = 2002
# Most-recent slice held out for the temporal backtest (strictly chronological split).
TEST_YEARS = 3

# The canonical model feature list. Order is contractual: the trained model stores it,
# and inference rebuilds the row in exactly this order.
FEATURES = [
    "elo_diff",  # home Elo - away Elo (venue-neutral skill estimate)
    "fifa_points_diff",  # home FIFA points - away FIFA points, as of the match date (point-in-time)
    "form_home",  # points-per-game over last FORM_WINDOW matches
    "form_away",
    "gd_home",  # avg goal difference over last FORM_WINDOW matches
    "gd_away",
    "h2h_home_winrate",  # decayed head-to-head score-share, home perspective
    "neutral",  # 1 if neutral venue (most WC matches), else 0
    "is_host_home",  # 1 if a 2026 host nation is playing in its own country
    "days_rest_home",  # days since each side's previous match (capped)
    "days_rest_away",
    # Squad strength (Phase 3) — point-in-time EA FC / FIFA ratings, home − away. Each negates
    # under a home/away swap (so reverse_features just flips the sign, like elo_diff). Third-party
    # in-game-rating estimates, not official data. Appended last; everything is name-keyed.
    "squad_strength_diff",  # home best-XI mean OVR − away best-XI mean OVR
    "attack_vs_def",  # (home attack − away defence) − (away attack − home defence): line matchup
    "depth_diff",  # home − away mean OVR of squad players ranked 12–26 (bench depth)
    "star_power_diff",  # home − away mean OVR of the top-3 players
]

# The Phase-3 squad subset (kept separate so model.py can build a squad-only baseline and the
# with/without-squad ablation without re-listing the names).
SQUAD_FEATURES = ["squad_strength_diff", "attack_vs_def", "depth_diff", "star_power_diff"]

# --------------------------------------------------------------------------------------
# Elo model (chess-style, eloratings.net-inspired weighting)
# --------------------------------------------------------------------------------------
ELO_BASE = 1500.0
ELO_HOME_ADV = 75.0  # rating bonus for the home side in the EXPECTATION (zeroed if neutral)

# Tournament importance -> base K-factor (eloratings.net scheme).
ELO_K_WORLD_CUP = 60.0
ELO_K_CONTINENTAL = 50.0
ELO_K_QUALIFIER = 40.0
ELO_K_OTHER = 30.0
ELO_K_FRIENDLY = 20.0

# Continental *finals* (not their qualifiers) — matched as substrings, lower-cased.
_CONTINENTAL_FINALS = (
    "uefa euro",
    "copa américa",
    "copa america",
    "african cup of nations",
    "afc asian cup",
    "gold cup",
    "concacaf championship",
    "oceania nations cup",
    "confederations cup",
)


def tournament_k(tournament: str) -> float:
    """Map a tournament name to its Elo base K-factor.

    Higher-stakes matches move ratings more. Qualifiers and Nations League sit below
    the continental/World-Cup finals; friendlies move ratings the least.
    """
    n = (tournament or "").lower()
    if "friendly" in n:
        return ELO_K_FRIENDLY
    if "world cup" in n and "qualif" not in n:
        return ELO_K_WORLD_CUP
    if "qualif" in n or "nations league" in n:
        return ELO_K_QUALIFIER
    if any(tag in n for tag in _CONTINENTAL_FINALS):
        return ELO_K_CONTINENTAL
    return ELO_K_OTHER


def goal_difference_multiplier(margin: int) -> float:
    """eloratings.net goal-difference weighting: bigger wins move ratings more."""
    margin = abs(int(margin))
    if margin <= 1:
        return 1.0
    if margin == 2:
        return 1.5
    return (11.0 + margin) / 8.0


def is_finals_tournament(tournament: str) -> bool:
    """True for World Cup / continental *finals* (not their qualifiers, not friendlies).

    Used to define ``is_host_home``: a home side playing in its own country at a major
    finals — the 2026-host situation. Defining it this way (rather than as plain
    ``1 - neutral``) keeps the feature from being perfectly collinear with ``neutral``.
    """
    n = (tournament or "").lower()
    if "world cup" in n and "qualif" not in n:
        return True
    return any(tag in n for tag in _CONTINENTAL_FINALS)


# --------------------------------------------------------------------------------------
# Team-name normalization
# --------------------------------------------------------------------------------------
# The results dataset, the 2026 draw, and (later) the squad data spell some nations
# differently. We normalize EVERYTHING through this map to a single canonical spelling,
# so Elo history, fixtures, and squads all join cleanly. Keys are matched case-insensitively
# after stripping whitespace; values are the canonical names used throughout the project
# (chosen to match the martj42 results dataset where possible).
TEAM_ALIASES = {
    "korea republic": "South Korea",
    "korea dpr": "North Korea",
    "republic of korea": "South Korea",
    "czech republic": "Czechia",
    "türkiye": "Turkey",
    "turkiye": "Turkey",
    "côte d'ivoire": "Ivory Coast",
    "cote d'ivoire": "Ivory Coast",
    "cabo verde": "Cape Verde",
    "cape verde islands": "Cape Verde",
    "ir iran": "Iran",
    "iran (islamic republic of)": "Iran",
    "congo dr": "DR Congo",
    "democratic republic of the congo": "DR Congo",
    "dr congo": "DR Congo",
    "usa": "United States",
    "united states of america": "United States",
    "bosnia & herzegovina": "Bosnia and Herzegovina",
    "china pr": "China",
    "chinese taipei": "Taiwan",
    "curacao": "Curaçao",
    "st kitts and nevis": "Saint Kitts and Nevis",
}


def normalize_team(name: str) -> str:
    """Return the canonical spelling of a national team name.

    Idempotent: canonical names map to themselves. Unknown names pass through unchanged
    (titles are not forced — the dataset already uses sensible capitalization).
    """
    if name is None:
        return name
    key = str(name).strip()
    return TEAM_ALIASES.get(key.lower(), key)
