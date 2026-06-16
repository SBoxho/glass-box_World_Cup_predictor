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
RAW_RESULTS_PATH = RAW_DIR / "results.csv"
WC2026_LIVE_CACHE_PATH = RAW_DIR / "wc2026_live.json"  # cached live-results snapshot (gitignored)
MATCHES_PATH = PROCESSED_DIR / "matches.parquet"
FEATURES_PATH = PROCESSED_DIR / "features.parquet"
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
    "form_home",  # points-per-game over last FORM_WINDOW matches
    "form_away",
    "gd_home",  # avg goal difference over last FORM_WINDOW matches
    "gd_away",
    "h2h_home_winrate",  # decayed head-to-head score-share, home perspective
    "neutral",  # 1 if neutral venue (most WC matches), else 0
    "is_host_home",  # 1 if a 2026 host nation is playing in its own country
    "days_rest_home",  # days since each side's previous match (capped)
    "days_rest_away",
]

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
