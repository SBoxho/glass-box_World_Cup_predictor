"""Model layer: train, calibrate, evaluate, persist, load, and predict.

The estimator is a multiclass XGBoost ({H, D, A}) wrapped in isotonic calibration so the
output numbers are usable as real probabilities, not just argmax labels. Validation is a
strict chronological split — never random K-fold — and is reported against two baselines
(always-home and Elo-only) so the ML lift is visible.

Two models are produced from one training call:
  * a *backtest* model (fit on older data, evaluated on a held-out recent slice) → metrics
  * a *production* model (refit on all available data, isotonic-calibrated via CV) → saved

Predictions for neutral venues are symmetrized: ``predict(A, B)`` averages both team orderings
so it mirrors ``predict(B, A)`` exactly (see :func:`predict_from_features`).
"""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, log_loss
from xgboost import XGBClassifier

from core import config
from core.config import CLASS_TO_IDX, CLASSES, FEATURES, SEED, SQUAD_FEATURES

XGB_PARAMS = {
    "n_estimators": 350,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "min_child_weight": 5.0,
    "reg_lambda": 2.0,
    "gamma": 0.5,
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "random_state": SEED,
    "n_jobs": 0,
    "verbosity": 0,
}


@dataclass
class ModelArtifact:
    """Everything the app/api need to make and explain a prediction, minus the live data."""

    model: CalibratedClassifierCV
    raw_model: XGBClassifier  # the underlying (uncalibrated) tree model — used by SHAP
    features: list[str]
    classes: list[str]
    trained_through: str
    metrics: dict


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _encode_y(result: pd.Series) -> np.ndarray:
    return result.map(CLASS_TO_IDX).to_numpy()


def _make_xgb() -> XGBClassifier:
    return XGBClassifier(**XGB_PARAMS)


CALIB_FOLDS = 5


def _fit_calibrated(
    df: pd.DataFrame, cols: list[str] | None = None
) -> tuple[CalibratedClassifierCV, XGBClassifier]:
    """Fit a cross-validated isotonic-calibrated XGBoost, plus a raw tree model for SHAP.

    Calibration uses ``CALIB_FOLDS``-fold internal CV (out-of-fold predictions feed the isotonic
    fit), which gives the calibrator far more data than a thin prefit slice would — important for
    a 3-class problem. The separately-fit raw XGBoost (trained on all of ``df``) is what
    :mod:`core.explain` runs SHAP on; calibration is a monotonic remap that does not change the
    sign/ranking of feature contributions. ``cols`` defaults to :data:`FEATURES`; the ablation
    passes a reduced column set.
    """
    cols = cols or FEATURES
    y = _encode_y(df["result"])
    calibrated = CalibratedClassifierCV(_make_xgb(), method="isotonic", cv=CALIB_FOLDS)
    calibrated.fit(df[cols], y)
    raw = _make_xgb()
    raw.fit(df[cols], y)
    return calibrated, raw


def temporal_split(
    df: pd.DataFrame, test_years: int = config.TEST_YEARS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological train/test split (no shuffling, ever).

    ``test`` is the most recent ``test_years`` of matches; everything earlier is ``trainval``
    (used for both the model and its internal calibration folds). The test slice is never seen
    during fitting or calibration.
    """
    df = df.sort_values("date", kind="mergesort").reset_index(drop=True)
    cutoff = df["date"].max() - pd.DateOffset(years=test_years)
    trainval = df[df["date"] <= cutoff].reset_index(drop=True)
    test = df[df["date"] > cutoff].reset_index(drop=True)
    return trainval, test


# --------------------------------------------------------------------------------------
# Baselines (so the ML lift is honest and visible)
# --------------------------------------------------------------------------------------
def _baseline_always_home(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Constant predictor: class probabilities = train marginals; label = home win."""
    freqs = train["result"].value_counts(normalize=True)
    probs = np.array([[freqs.get(c, 0.0) for c in CLASSES]] * len(test))
    y = _encode_y(test["result"])
    return {
        "logloss": float(log_loss(y, probs, labels=[0, 1, 2])),
        "accuracy": float((test["result"] == "H").mean()),
    }


def _baseline_logreg(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> dict:
    """Multinomial logistic regression on ``cols`` only — a 'single strength signal' bar."""
    lr = LogisticRegression(max_iter=1000, C=1.0)
    lr.fit(train[cols], _encode_y(train["result"]))
    probs = lr.predict_proba(test[cols])
    # Align to CLASSES order regardless of the encoder's internal class order.
    probs = probs[:, [list(lr.classes_).index(i) for i in range(len(CLASSES))]]
    y = _encode_y(test["result"])
    return {
        "logloss": float(log_loss(y, probs, labels=[0, 1, 2])),
        "accuracy": float(accuracy_score(y, probs.argmax(1))),
    }


def _baseline_elo_only(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Logistic regression on Elo gap + venue only — the original 'no ML features' bar."""
    return _baseline_logreg(train, test, ["elo_diff", "neutral"])


def _baseline_fifa_only(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Logistic regression on FIFA-points gap + venue only — the third baseline (vs Elo-only)."""
    return _baseline_logreg(train, test, ["fifa_points_diff", "neutral"])


def _baseline_squad_only(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Logistic regression on the squad-strength diffs + venue only — the squad 'single signal' bar."""
    return _baseline_logreg(train, test, [*SQUAD_FEATURES, "neutral"])


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------
def reliability_curve(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> dict:
    """Pooled one-vs-rest reliability: bin every predicted class-probability, compare to the
    observed frequency in each bin. Returns bin centers, observed freq, and ECE."""
    onehot = np.eye(len(CLASSES))[y_true]
    p = proba.ravel()
    hit = onehot.ravel()
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    mean_pred, obs_freq, weight = [], [], []
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        mean_pred.append(float(p[mask].mean()))
        obs_freq.append(float(hit[mask].mean()))
        weight.append(int(mask.sum()))
    w = np.array(weight, dtype=float)
    ece = float(np.sum(w * np.abs(np.array(mean_pred) - np.array(obs_freq))) / w.sum())
    return {"mean_pred": mean_pred, "obs_freq": obs_freq, "weight": weight, "ece": ece}


def evaluate(
    model: CalibratedClassifierCV, test: pd.DataFrame, cols: list[str] | None = None
) -> dict:
    """Compute log-loss, accuracy, per-class P/R/F1, and the reliability curve on the test set.

    ``cols`` defaults to :data:`FEATURES`; the ablation evaluates a model fit on a reduced set.
    """
    cols = cols or FEATURES
    y = _encode_y(test["result"])
    proba = model.predict_proba(test[cols])
    proba = proba[:, [list(model.classes_).index(i) for i in range(len(CLASSES))]]
    report = classification_report(
        y,
        proba.argmax(1),
        labels=[0, 1, 2],
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )
    return {
        "n_test": int(len(test)),
        "logloss": float(log_loss(y, proba, labels=[0, 1, 2])),
        "accuracy": float(accuracy_score(y, proba.argmax(1))),
        "per_class": {c: report[c] for c in CLASSES},
        "reliability": reliability_curve(y, proba),
    }


# --------------------------------------------------------------------------------------
# Training entry point
# --------------------------------------------------------------------------------------
def _safe_corr(a: pd.Series, b: pd.Series) -> float | None:
    """Pearson r, or ``None`` if undefined (a constant column → NaN, which isn't valid JSON)."""
    r = a.corr(b)
    return float(r) if pd.notna(r) else None


def train_model(features_df: pd.DataFrame) -> ModelArtifact:
    """Run the temporal backtest and fit the production model; return a saveable artifact."""
    trainval, test = temporal_split(features_df)

    # 1) Honest backtest: this model (and its calibration folds) never see the test slice.
    backtest_model, _ = _fit_calibrated(trainval)
    model_metrics = evaluate(backtest_model, test)

    # 1b) Ablation: refit the same backtest on FEATURES minus the squad features and compare, so the
    # squad-feature contribution is visible (expected to be small — squad strength ≈ collinear with
    # Elo). The "with_squad" numbers are exactly the full backtest model above.
    non_squad = [f for f in FEATURES if f not in SQUAD_FEATURES]
    ablation_model, _ = _fit_calibrated(trainval, cols=non_squad)
    without_squad = evaluate(ablation_model, test, cols=non_squad)

    metrics = {
        "split": {
            "trainval": [
                str(trainval["date"].min().date()),
                str(trainval["date"].max().date()),
                len(trainval),
            ],
            "test": [
                str(test["date"].min().date()),
                str(test["date"].max().date()),
                len(test),
            ],
        },
        "model": model_metrics,
        "baselines": {
            "always_home": _baseline_always_home(trainval, test),
            "elo_only": _baseline_elo_only(trainval, test),
            "fifa_only": _baseline_fifa_only(trainval, test),
            "squad_only": _baseline_squad_only(trainval, test),
        },
        "ablation": {
            "squad_features": SQUAD_FEATURES,
            "with_squad": {
                "logloss": model_metrics["logloss"],
                "accuracy": model_metrics["accuracy"],
                "ece": model_metrics["reliability"]["ece"],
            },
            "without_squad": {
                "logloss": without_squad["logloss"],
                "accuracy": without_squad["accuracy"],
                "ece": without_squad["reliability"]["ece"],
            },
            # Positive delta_logloss / delta_accuracy ⇒ adding the squad features *helped*.
            "delta_logloss": float(without_squad["logloss"] - model_metrics["logloss"]),
            "delta_accuracy": float(model_metrics["accuracy"] - without_squad["accuracy"]),
        },
        "features": FEATURES,
        "feature_notes": {
            # Elo, FIFA points and squad strength measure overlapping things — report the
            # collinearity honestly (this is why the squad lift is expected to be small/null).
            "elo_fifa_pearson": _safe_corr(
                features_df["elo_diff"], features_df["fifa_points_diff"]
            ),
            "elo_squad_pearson": _safe_corr(
                features_df["elo_diff"], features_df["squad_strength_diff"]
            ),
        },
    }

    # 2) Production model + raw SHAP model: refit on ALL available data.
    full = features_df.sort_values("date", kind="mergesort").reset_index(drop=True)
    prod_model, prod_raw = _fit_calibrated(full)

    return ModelArtifact(
        model=prod_model,
        raw_model=prod_raw,
        features=FEATURES,
        classes=CLASSES,
        trained_through=str(full["date"].max().date()),
        metrics=metrics,
    )


# --------------------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------------------
def save_model(artifact: ModelArtifact, path=None) -> None:
    path = path or config.MODEL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path, compress=3)  # tree models compress well — keeps the commit small


def load_model(path=None) -> ModelArtifact:
    path = path or config.MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run scripts/train.py first.")
    return joblib.load(path)


# --------------------------------------------------------------------------------------
# Prediction (with neutral-venue symmetry)
# --------------------------------------------------------------------------------------
def reverse_features(f: dict) -> dict:
    """Swap home/away in a feature dict (the mirror used for neutral-venue symmetrization)."""
    return {
        "elo_diff": -f["elo_diff"],
        "fifa_points_diff": -f["fifa_points_diff"],
        "form_home": f["form_away"],
        "form_away": f["form_home"],
        "gd_home": f["gd_away"],
        "gd_away": f["gd_home"],
        "h2h_home_winrate": 1.0 - f["h2h_home_winrate"],
        "neutral": f["neutral"],
        "is_host_home": f["is_host_home"],
        "days_rest_home": f["days_rest_away"],
        "days_rest_away": f["days_rest_home"],
        # Squad diffs are all antisymmetric (home − away), so the mirror just flips the sign.
        "squad_strength_diff": -f["squad_strength_diff"],
        "attack_vs_def": -f["attack_vs_def"],
        "depth_diff": -f["depth_diff"],
        "star_power_diff": -f["star_power_diff"],
    }


def _proba(model, feats: dict, cols: list[str]) -> np.ndarray:
    x = np.array([[feats[c] for c in cols]], dtype=float)
    p = model.predict_proba(x)[0]
    order = [list(model.classes_).index(i) for i in range(len(CLASSES))]
    return p[order]


def predict_from_features(
    artifact: ModelArtifact, feats: dict, neutral: bool | None = None
) -> dict[str, float]:
    """Probabilities {H, D, A} for a single feature row.

    For neutral venues the prediction is the average of both team orderings (with H/A swapped),
    so it is exactly mirror-symmetric — there is no spurious 'home' advantage at a neutral site.
    """
    if neutral is None:
        neutral = bool(feats["neutral"])
    p = _proba(artifact.model, feats, artifact.features)
    if neutral:
        rev = _proba(artifact.model, reverse_features(feats), artifact.features)
        p = 0.5 * (p + rev[::-1])  # rev[::-1] swaps H<->A, leaves D
    p = p / p.sum()
    return dict(zip(CLASSES, (float(v) for v in p), strict=True))


class Predictor:
    """High-level predictor: a trained artifact + a live :class:`InferenceState`.

    This is the single object the app, the API, and the simulator all call. ``core`` stays
    UI-agnostic; the caller is responsible for building the inference state from match history.
    """

    def __init__(self, artifact: ModelArtifact, state):
        self.artifact = artifact
        self.state = state

    def features(self, home, away, neutral=True, is_host_home=None, ref_date=None) -> dict:
        if is_host_home is None:
            is_host_home = int(not neutral)
        return self.state.feature_row(home, away, neutral, is_host_home, ref_date)

    def predict(
        self, home, away, neutral=True, is_host_home=None, ref_date=None
    ) -> dict[str, float]:
        feats = self.features(home, away, neutral, is_host_home, ref_date)
        return predict_from_features(self.artifact, feats, neutral=neutral)

    def predict_pairs_neutral(self, pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
        """Batched, symmetrized neutral-venue predictions for many pairs in one model call.

        Used by the tournament simulator to build its pairwise matrix quickly — a single
        ``predict_proba`` over all rows is far faster than one model call per pair.
        """
        cols = self.artifact.features
        rows = []
        for a, b in pairs:
            f = self.state.feature_row(a, b, neutral=True, is_host_home=0)
            rows.append([f[c] for c in cols])
            rows.append([reverse_features(f)[c] for c in cols])
        proba = self.artifact.model.predict_proba(np.array(rows, dtype=float))
        order = [list(self.artifact.model.classes_).index(i) for i in range(len(CLASSES))]
        proba = proba[:, order]
        out = []
        for k in range(len(pairs)):
            p = 0.5 * (proba[2 * k] + proba[2 * k + 1][::-1])
            p = p / p.sum()
            out.append(dict(zip(CLASSES, (float(v) for v in p), strict=True)))
        return out
