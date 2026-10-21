"""Train the pre-tournament replay model (no 2026 World Cup match in its training data).

    python scripts/train_pre_wc.py

Same recipe as ``scripts/train.py`` (temporal backtest + calibrated XGBoost refit on all rows), but
on the features of matches strictly *before* the 2026 opener (:data:`core.replay.REPLAY_CUTOFF`).
Features are point-in-time (guarded by ``tests/test_features_no_leakage.py``), so dropping the later
rows is exactly equivalent to building the dataset on pre-tournament history alone.

Writes only ``models/model_pre_wc.joblib`` — the production model, metrics and plots are untouched.
The Simulate tab uses this artifact for its post-tournament replay (see ``core/replay.py``).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from core import config, model, replay  # noqa: E402


def main() -> None:
    if not config.FEATURES_PATH.exists():
        sys.exit("features.parquet not found — run scripts/build_dataset.py first.")
    feats = pd.read_parquet(config.FEATURES_PATH)
    pre = feats[feats["date"] < replay.REPLAY_CUTOFF].reset_index(drop=True)
    print(
        f"Training on {len(pre):,} matches ({pre['date'].min().date()} -> {pre['date'].max().date()}); "
        f"dropped {len(feats) - len(pre)} on/after {replay.REPLAY_CUTOFF.date()}."
    )

    artifact = model.train_model(pre)
    assert pd.Timestamp(artifact.trained_through) < replay.REPLAY_CUTOFF
    model.save_model(artifact, config.REPLAY_MODEL_PATH)

    m = artifact.metrics["model"]
    print(
        f"Backtest: logloss={m['logloss']:.4f}  acc={m['accuracy']:.3f}  "
        f"ECE={m['reliability']['ece']:.3f}"
    )
    print(f"Saved {config.REPLAY_MODEL_PATH.name} (trained through {artifact.trained_through}).")


if __name__ == "__main__":
    main()
