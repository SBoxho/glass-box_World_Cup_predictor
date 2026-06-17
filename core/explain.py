"""Explainability — the headline of the project.

Wraps ``shap.TreeExplainer`` over the raw XGBoost model to produce:

* per-match feature contributions toward the favoured outcome, plus an auto-generated
  plain-English narrative ("Model leans Brazil: +Elo gap, +form; tempered by neutral venue");
* a global SHAP summary saved as a figure for the "Under the Hood" tab.

SHAP runs on the *raw* (uncalibrated) tree model. Calibration is a monotonic remap of the
probabilities, so it does not change which features push toward an outcome or their sign — it
only rescales the final numbers. This caveat is stated in the app.
"""

from __future__ import annotations

import numpy as np

from core.config import CLASS_TO_IDX, CLASSES, FEATURES

# Human-readable labels for the model features (used in narratives and charts).
FEATURE_LABELS = {
    "elo_diff": "Elo rating gap",
    "form_home": "home recent form",
    "form_away": "away recent form",
    "gd_home": "home scoring form",
    "gd_away": "away scoring form",
    "h2h_home_winrate": "head-to-head record",
    "neutral": "neutral venue",
    "is_host_home": "host-nation advantage",
    "days_rest_home": "home days of rest",
    "days_rest_away": "away days of rest",
    # Squad strength (EA FC ratings — third-party estimates). Generic (no "home"/"away" substring,
    # like "Elo rating gap") so _team_label leaves them untouched.
    "squad_strength_diff": "squad strength gap (EA FC)",
    "attack_vs_def": "attack vs defence (EA FC)",
    "depth_diff": "squad depth gap (EA FC)",
    "star_power_diff": "star power gap (EA FC)",
}


def build_explainer(artifact):
    """Create a TreeExplainer for the artifact's raw model (cheap; cache at the call site)."""
    import shap

    return shap.TreeExplainer(artifact.raw_model)


def _team_label(feature: str, home: str, away: str) -> str:
    """Replace the generic 'home'/'away' in a feature label with the actual team names."""
    label = FEATURE_LABELS.get(feature, feature)
    return label.replace("home", home).replace("away", away)


def explain_match(artifact, feats: dict, probs: dict, home: str, away: str, explainer=None) -> dict:
    """Explain a single prediction: contributions toward the favoured side + a narrative.

    Returns a dict with the favoured team, the class explained, a sorted contribution list
    (each: feature, label, signed SHAP toward the favoured side, raw value), the SHAP base
    value, and a plain-English ``narrative`` string.
    """
    explainer = explainer or build_explainer(artifact)
    x = np.array([[feats[c] for c in FEATURES]], dtype=float)
    sv = explainer.shap_values(x)  # shape (1, n_features, n_classes)
    base = np.atleast_1d(explainer.expected_value)

    # Favoured side from the (calibrated) probabilities: whoever is likelier to win.
    favored, other = (home, away) if probs["H"] >= probs["A"] else (away, home)
    cls = "H" if favored == home else "A"
    c = CLASS_TO_IDX[cls]
    win_prob = probs[cls]

    contribs = []
    for i, feat in enumerate(FEATURES):
        contribs.append(
            {
                "feature": feat,
                "label": _team_label(feat, home, away),
                "shap": float(sv[0, i, c]),  # >0 pushes toward the favoured side
                "value": float(feats[feat]),
            }
        )
    contribs.sort(key=lambda d: abs(d["shap"]), reverse=True)

    narrative = _narrate(favored, other, win_prob, probs["D"], contribs)
    return {
        "favored": favored,
        "underdog": other,
        "class": cls,
        "win_prob": win_prob,
        "draw_prob": probs["D"],
        "base_value": float(base[c]),
        "contributions": contribs,
        "narrative": narrative,
    }


def _narrate(favored: str, other: str, win_prob: float, draw_prob: float, contribs: list) -> str:
    """Compose a one-paragraph plain-English explanation from the top ± contributors."""
    pos = [d for d in contribs if d["shap"] > 1e-6][:3]
    neg = [d for d in contribs if d["shap"] < -1e-6][:2]

    lead = f"The model leans **{favored}** ({win_prob:.0%} to win, {draw_prob:.0%} draw). "
    if pos:
        lead += "Pushing that way: " + ", ".join(d["label"] for d in pos) + "."
    else:
        lead += "It is close to a coin flip on the modelled factors."
    if neg:
        lead += f" Working against {favored}: " + ", ".join(d["label"] for d in neg) + "."
    return lead


def save_global_summary(artifact, features_df, path, sample: int = 2000, seed: int = 0) -> None:
    """Render and save a global SHAP feature-importance summary (mean |SHAP| per class)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    df = features_df
    if len(df) > sample:
        df = df.sample(sample, random_state=seed)
    x = df[FEATURES]

    explainer = shap.TreeExplainer(artifact.raw_model)
    sv = explainer.shap_values(x)  # (n, n_features, n_classes)

    # Mean |SHAP| per feature per class -> grouped horizontal bars.
    importance = np.abs(sv).mean(axis=0)  # (n_features, n_classes)
    order = np.argsort(importance.sum(axis=1))
    labels = [FEATURE_LABELS.get(FEATURES[i], FEATURES[i]) for i in order]

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    y = np.arange(len(order))
    colors = {"H": "#2563eb", "D": "#9ca3af", "A": "#ef4444"}
    left = np.zeros(len(order))
    for ci, cname in enumerate(CLASSES):
        vals = importance[order, ci]
        ax.barh(y, vals, left=left, color=colors[cname], label=f"{cname}")
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("mean |SHAP| (impact on model output)")
    ax.set_title("Global feature importance (SHAP)\nstacked by outcome class H / D / A")
    ax.legend(title="class", loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
