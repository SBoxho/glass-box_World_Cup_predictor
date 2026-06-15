"""Train + calibrate the model and write the committed artifacts.

    python scripts/train.py

Reads data/processed/features.parquet (run scripts/build_dataset.py first), then writes:
    models/model.joblib       calibrated production model + feature metadata + backtest metrics
    models/metrics.json       backtest metrics vs baselines (shown in the app's "Under the Hood")
    models/reliability.png    calibration / reliability curve on the temporal test set
    models/shap_summary.png   global SHAP feature-importance (beeswarm), if SHAP is available
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from core import config, features, ingest, model  # noqa: E402


def _load_features() -> pd.DataFrame:
    if config.FEATURES_PATH.exists():
        return pd.read_parquet(config.FEATURES_PATH)
    print("features.parquet not found — building it from match history first ...")
    matches = ingest.get_clean_matches()
    return features.build_features(matches, train_start_year=config.TRAIN_START_YEAR)


def _plot_reliability(metrics: dict, path: Path) -> None:
    rel = metrics["model"]["reliability"]
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="perfectly calibrated")
    ax.plot(rel["mean_pred"], rel["obs_freq"], "o-", color="#2563eb", label="model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(f"Reliability curve (test set)\nECE = {rel['ece']:.3f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_global_shap(artifact, features_df: pd.DataFrame, path: Path) -> None:
    """Render a global SHAP beeswarm over a sample of the data (best-effort)."""
    try:
        from core import explain

        explain.save_global_summary(artifact, features_df, path)
    except Exception as exc:  # pragma: no cover - plotting is non-critical to training
        print(f"  (skipped global SHAP plot: {exc})")


def main() -> None:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    feats = _load_features()
    print(
        f"Training on {len(feats):,} matches ({feats['date'].min().date()} -> "
        f"{feats['date'].max().date()})."
    )

    artifact = model.train_model(feats)
    model.save_model(artifact)
    config.METRICS_PATH.write_text(json.dumps(artifact.metrics, indent=2), encoding="utf-8")
    _plot_reliability(artifact.metrics, config.RELIABILITY_PLOT_PATH)
    _plot_global_shap(artifact, feats, config.SHAP_SUMMARY_PATH)

    m = artifact.metrics
    sp = m["split"]
    print("\nTemporal split:")
    print(f"  trainval {sp['trainval'][0]}..{sp['trainval'][1]} ({sp['trainval'][2]:,})")
    print(f"  test     {sp['test'][0]}..{sp['test'][1]} ({sp['test'][2]:,})")
    print("\nTest-set performance (lower log-loss is better):")
    print(
        f"  {'model (calibrated XGB)':<26} logloss={m['model']['logloss']:.4f}  "
        f"acc={m['model']['accuracy']:.3f}  ECE={m['model']['reliability']['ece']:.3f}"
    )
    print(
        f"  {'baseline: Elo-only':<26} logloss={m['baselines']['elo_only']['logloss']:.4f}  "
        f"acc={m['baselines']['elo_only']['accuracy']:.3f}"
    )
    print(
        f"  {'baseline: always-home':<26} logloss={m['baselines']['always_home']['logloss']:.4f}  "
        f"acc={m['baselines']['always_home']['accuracy']:.3f}"
    )
    print(
        f"\nSaved: {config.MODEL_PATH.name}, {config.METRICS_PATH.name}, "
        f"{config.RELIABILITY_PLOT_PATH.name}"
    )


if __name__ == "__main__":
    main()
