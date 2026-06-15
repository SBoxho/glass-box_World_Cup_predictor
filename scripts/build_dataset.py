"""Build the processed dataset: download + clean match history, then engineer features.

    python scripts/build_dataset.py [--force]

Outputs (gitignored, regenerable):
    data/processed/matches.parquet   cleaned full match history (Elo/inference warm-up)
    data/processed/features.parquet  point-in-time training table (filtered to the modern era)

Run this before scripts/train.py.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow running as a plain script (python scripts/build_dataset.py) without installing the pkg.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import config, features, ingest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the processed World Cup dataset.")
    parser.add_argument("--force", action="store_true", help="re-download results.csv")
    args = parser.parse_args()

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print("Downloading + cleaning match history ...")
    matches = ingest.get_clean_matches(force_download=args.force)
    matches.to_parquet(config.MATCHES_PATH, index=False)
    print(
        f"  {len(matches):,} matches "
        f"({matches['date'].min().date()} -> {matches['date'].max().date()}) "
        f"-> {config.MATCHES_PATH.relative_to(config.BASE_DIR)}"
    )

    print("Engineering point-in-time features ...")
    feats = features.build_features(matches, train_start_year=config.TRAIN_START_YEAR)
    feats.to_parquet(config.FEATURES_PATH, index=False)
    print(
        f"  {len(feats):,} training rows x {len(config.FEATURES)} features "
        f"(>= {config.TRAIN_START_YEAR}) -> {config.FEATURES_PATH.relative_to(config.BASE_DIR)}"
    )

    dist = feats["result"].value_counts(normalize=True).round(3).to_dict()
    print(f"  outcome distribution (H/D/A): {dist}")
    print(f"Done in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
