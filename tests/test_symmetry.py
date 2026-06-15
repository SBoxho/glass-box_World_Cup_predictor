"""Guardrail 2 — neutral-venue symmetry.

At a neutral venue there is no real 'home' side, so ``predict(A, B)`` must be the mirror image
of ``predict(B, A)``: P(A beats B) and P(B beats A) swap, P(draw) is unchanged. The model
guarantees this by averaging both team orderings at inference (see
:func:`core.model.predict_from_features`).

We also check the *opposite*: with a real home advantage (non-neutral) the prediction is NOT
forced symmetric — otherwise the symmetrization would be hiding a broken venue feature.
"""

from __future__ import annotations

import pytest
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

from core import model
from core.config import CLASS_TO_IDX, CLASSES, FEATURES
from core.features import build_features, build_inference_state


def _tiny_artifact(feats) -> model.ModelArtifact:
    """A small, fast calibrated model — symmetry is structural, so model size is irrelevant."""
    params = dict(model.XGB_PARAMS)
    params.update(n_estimators=60, max_depth=3)
    y = feats["result"].map(CLASS_TO_IDX)
    cal = CalibratedClassifierCV(XGBClassifier(**params), method="isotonic", cv=3)
    cal.fit(feats[FEATURES], y)
    raw = XGBClassifier(**params)
    raw.fit(feats[FEATURES], y)
    return model.ModelArtifact(
        model=cal,
        raw_model=raw,
        features=FEATURES,
        classes=CLASSES,
        trained_through="test",
        metrics={},
    )


@pytest.fixture(scope="module")
def predictor(synthetic_matches):
    feats = build_features(synthetic_matches)
    state = build_inference_state(synthetic_matches)
    return model.Predictor(_tiny_artifact(feats), state)


PAIRS = [("Team00", "Team09"), ("Team03", "Team11"), ("Team05", "Team14"), ("Team01", "Team07")]


def test_neutral_predictions_are_mirror_symmetric(predictor):
    for a, b in PAIRS:
        p = predictor.predict(a, b, neutral=True)
        q = predictor.predict(b, a, neutral=True)
        assert p["H"] == pytest.approx(q["A"], abs=1e-6)
        assert p["A"] == pytest.approx(q["H"], abs=1e-6)
        assert p["D"] == pytest.approx(q["D"], abs=1e-6)
        assert sum(p.values()) == pytest.approx(1.0, abs=1e-9)


def test_home_advantage_breaks_symmetry(predictor):
    """A non-neutral match must NOT be symmetrized away: predict(A,B) and predict(B,A) at a
    real (non-neutral) venue should differ, because the home side carries an advantage.

    (We assert asymmetry, not the *direction* per pair — the sign of a single pair's home boost
    can wobble for the tiny synthetic model across platforms; the directional 'home helps' claim
    belongs to the real trained model, not this hermetic structural check.)"""
    differences = [
        abs(
            predictor.predict(a, b, neutral=False)["H"]
            - predictor.predict(b, a, neutral=False)["A"]
        )
        for a, b in PAIRS
    ]
    assert max(differences) > 1e-3, "non-neutral predictions look symmetric — venue feature dead?"
