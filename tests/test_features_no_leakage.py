"""Guardrail 1 — no leakage.

For sampled matches, the features produced by the batch pipeline must be reproducible from a
history truncated to strictly before the match date. If any feature peeked at same-day or
future matches, the independent recompute would disagree.

We also assert the boundary is *strict*: shifting the as-of date forward by a day (which lets
the match's own day leak in) must change the features — proof the strict-before cut is real and
not a no-op.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core import config
from core.features import build_features, features_for_match

# Historically-derived features (the ones that could leak). neutral / is_host_home are
# same-row metadata, not derived from other matches.
_DERIVED = [
    "elo_diff",
    "form_home",
    "form_away",
    "gd_home",
    "gd_away",
    "h2h_home_winrate",
    "days_rest_home",
    "days_rest_away",
]


def test_pipeline_matches_strict_before_recompute(synthetic_matches):
    feats = build_features(synthetic_matches)

    # Sample from the back half so there is real history behind each match.
    rng = np.random.default_rng(123)
    candidates = feats.index[len(feats) // 2 :]
    sample = rng.choice(candidates, size=25, replace=False)

    for i in sample:
        row = feats.loc[i]
        recomputed = features_for_match(
            synthetic_matches,
            home=row["home_team"],
            away=row["away_team"],
            date=row["date"],
            neutral=bool(row["neutral"]),
            is_host_home=int(row["is_host_home"]),
        )
        for feat in config.FEATURES:
            assert recomputed[feat] == _approx(row[feat]), (
                f"{feat} mismatch for {row['home_team']} vs {row['away_team']} "
                f"on {row['date'].date()}: pipeline={row[feat]} recompute={recomputed[feat]}"
            )


def test_boundary_is_strictly_before(synthetic_matches):
    """Letting the match's own day into history must move at least one derived feature."""
    feats = build_features(synthetic_matches)
    # Find a match that shares its date with at least one other match (so 'same day' has bite).
    counts = feats["date"].value_counts()
    busy_dates = set(counts[counts > 1].index)
    row = feats[feats["date"].isin(busy_dates)].iloc[-1]

    strict = features_for_match(
        synthetic_matches,
        row["home_team"],
        row["away_team"],
        row["date"],
        bool(row["neutral"]),
        int(row["is_host_home"]),
    )
    leaked = features_for_match(
        synthetic_matches,
        row["home_team"],
        row["away_team"],
        row["date"] + pd.Timedelta(days=1),  # now includes the match's own day
        bool(row["neutral"]),
        int(row["is_host_home"]),
    )
    assert any(strict[f] != _approx(leaked[f]) for f in _DERIVED), (
        "Including the match's own day changed nothing — the strict-before cut may be a no-op."
    )


class _approx:
    """Tiny float-tolerant comparison helper (avoids a pytest.approx import dance)."""

    def __init__(self, value, rel=1e-9, abs_=1e-9):
        self.value = float(value)
        self.rel = rel
        self.abs = abs_

    def __eq__(self, other):
        return abs(float(other) - self.value) <= max(self.abs, self.rel * abs(self.value))

    def __repr__(self):
        return f"~{self.value}"
