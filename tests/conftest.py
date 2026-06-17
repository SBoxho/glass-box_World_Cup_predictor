"""Shared test fixtures.

The guardrail tests are deliberately hermetic: they synthesize a small, deterministic match
history and (where needed) train a tiny model on it. No network, no committed artifacts — so
CI is fast and reproducible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

TEAMS = [f"Team{i:02d}" for i in range(16)]
TOURNAMENTS = ["Friendly", "FIFA World Cup qualification", "UEFA Euro", "FIFA World Cup"]


def make_synthetic_matches(seed: int = 0, n_days: int = 500) -> pd.DataFrame:
    """Generate a cleaned-schema match frame from latent team strengths + Poisson scores.

    Intentionally includes multiple matches on the same day so the date-batched, no-leakage
    code paths are actually exercised.
    """
    rng = np.random.default_rng(seed)
    strength = {t: rng.normal(0.0, 1.0) for t in TEAMS}
    rows = []
    cur = pd.Timestamp("2008-01-01")
    for _ in range(n_days):
        cur = cur + pd.Timedelta(days=int(rng.integers(3, 12)))
        for _ in range(int(rng.integers(1, 4))):  # 1-3 matches per active day
            h, a = (str(x) for x in rng.choice(TEAMS, size=2, replace=False))
            neutral = bool(rng.random() < 0.3)
            adv = 0.0 if neutral else 0.4
            lam_h = max(0.2, 1.3 + strength[h] - strength[a] + adv)
            lam_a = max(0.2, 1.3 + strength[a] - strength[h])
            rows.append(
                {
                    "date": cur,
                    "home_team": h,
                    "away_team": a,
                    "home_score": int(rng.poisson(lam_h)),
                    "away_score": int(rng.poisson(lam_a)),
                    "tournament": str(rng.choice(TOURNAMENTS, p=[0.6, 0.25, 0.1, 0.05])),
                    "country": "Neutralia",
                    "neutral": neutral,
                }
            )

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["margin"] = df["home_score"] - df["away_score"]
    df["result"] = np.where(df["margin"] > 0, "H", np.where(df["margin"] < 0, "A", "D"))
    df = df.sort_values("date", kind="mergesort").reset_index(drop=True)
    df["year"] = df["date"].dt.year
    return df[
        [
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
    ]


@pytest.fixture(scope="session")
def synthetic_matches() -> pd.DataFrame:
    return make_synthetic_matches(seed=7, n_days=600)


def make_synthetic_rankings(seed: int = 11) -> pd.DataFrame:
    """Generate a tidy ``(team, total_points, date)`` ranking time series for the synthetic teams.

    Snapshots are spaced ~120 days apart across the match window so they interleave with matches
    (giving the point-in-time as-of join real bite), and each team's points drift over time, so
    the FIFA feature is non-constant and time-varying — exactly what the no-leakage test needs.
    """
    rng = np.random.default_rng(seed)
    base = {t: 1000.0 + 50.0 * i for i, t in enumerate(TEAMS)}  # distinct per team
    dates = pd.date_range("2008-02-01", "2020-12-01", freq="120D")
    rows = [
        {"team": t, "total_points": round(float(base[t] + rng.normal(0, 30)), 2), "date": d}
        for d in dates
        for t in TEAMS
    ]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date", kind="mergesort").reset_index(drop=True)


@pytest.fixture(scope="session")
def synthetic_rankings() -> pd.DataFrame:
    return make_synthetic_rankings(seed=11)


def make_synthetic_squads(seed: int = 13) -> pd.DataFrame:
    """Generate a time-indexed per-(team, date) squad-strength table for the synthetic teams.

    Columns mirror :data:`core.squads.COLUMNS` (team, date, the five OVR components). Versions are
    spaced ~200 days apart (sparser than the rankings, like ~yearly game releases) so they
    interleave with matches and give the point-in-time as-of join real bite, and each team's level
    drifts over time so the squad features are non-constant — exactly what the no-leakage test needs.
    """
    rng = np.random.default_rng(seed)
    level = {t: 70.0 + 0.8 * i for i, t in enumerate(TEAMS)}  # distinct base level per team
    dates = pd.date_range("2008-03-01", "2020-10-01", freq="200D")
    rows = []
    for d in dates:
        for t in TEAMS:
            lvl = level[t] + rng.normal(0, 1.5)
            rows.append(
                {
                    "team": t,
                    "date": d,
                    "bestxi_ovr": round(lvl, 2),
                    "attack_ovr": round(lvl + rng.normal(0, 1.0), 2),
                    "def_ovr": round(lvl + rng.normal(0, 1.0), 2),
                    "depth_ovr": round(lvl - 4.0 + rng.normal(0, 1.0), 2),
                    "star3_ovr": round(lvl + 5.0 + rng.normal(0, 1.0), 2),
                }
            )
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date", kind="mergesort").reset_index(drop=True)


@pytest.fixture(scope="session")
def synthetic_squads() -> pd.DataFrame:
    return make_synthetic_squads(seed=13)
