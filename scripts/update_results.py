"""Fetch live 2026 World Cup results and (optionally) lock them into data/wc2026.json.

    python scripts/update_results.py            # dry run: fetch + print, change nothing
    python scripts/update_results.py --write    # also bake known_results into data/wc2026.json

The default is a dry run so you can eyeball the upstream data before trusting it (the openfootball
feed is community-maintained — verify against official results before committing a snapshot).
``--write`` writes the played group results plus an "as of" timestamp into the committed tournament
file so a deployment starts from the live standings with no network call. It rewrites the JSON via
``json.dump`` (formatting may normalize). The Streamlit app can also pull live results on demand via
its "Refresh live results" button, so committing a snapshot is entirely optional.

Source: openfootball/worldcup.json (CC0 / public domain). See DATA_SOURCES.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as a plain script (python scripts/update_results.py) without installing the pkg.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import config, live  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch live 2026 World Cup group results.")
    parser.add_argument("--url", default=None, help="override the live-results URL")
    parser.add_argument(
        "--write",
        action="store_true",
        help="write known_results into data/wc2026.json (default: dry run, print only)",
    )
    args = parser.parse_args()

    print(f"Fetching live results from {args.url or config.WC2026_LIVE_URL} ...")
    snapshot = live.fetch_live_results(url=args.url)
    results = snapshot["known_results"]
    print(f"  source:     {snapshot['source']}")
    print(f"  fetched_at: {snapshot['fetched_at']}")
    print(f"  {len(results)} played group match(es):")
    for r in results:
        print(f"    {r['home']} {r['home_score']}-{r['away_score']} {r['away']}")
    if "error" in snapshot:
        print(f"  (note: fetch degraded — {snapshot['error']})")

    if not results:
        print("No played group results available; nothing to write.")
        return

    if not args.write:
        print("\nDry run — data/wc2026.json unchanged. Re-run with --write to bake these in.")
        return

    path = config.WC2026_PATH
    data = json.loads(path.read_text(encoding="utf-8"))
    data["known_results"] = results
    data["known_results_as_of"] = snapshot["fetched_at"]
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rel = path.relative_to(config.BASE_DIR)
    print(f"\nWrote {len(results)} result(s) + as-of timestamp to {rel}.")


if __name__ == "__main__":
    main()
