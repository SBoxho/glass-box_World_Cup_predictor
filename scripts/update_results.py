"""Refresh the committed 2026 data from the live openfootball feed.

Two independent, opt-in operations against ``data/wc2026.json`` — both default to a *dry run* so
you can eyeball the community-maintained upstream before trusting it:

    python scripts/update_results.py                     # dry run: print played group results
    python scripts/update_results.py --write             # bake known_results + as-of into the file
    python scripts/update_results.py --fixtures          # dry run: print the full schedule
    python scripts/update_results.py --fixtures --write  # bake the structure-only fixtures[] schedule

``known_results`` locks already-played group matches so the Tournament Simulator runs forward from
the live standings. ``fixtures[]`` is the *structure-only* schedule (teams/knockout placeholders,
group/round, venue, date, resolved UTC kick-off — no scores) that the Matchday Home page reads
offline; live scores remain an optional overlay merged on top at render time. Both bakes are
optional: the app can also pull live data on demand.

Source: openfootball/worldcup.json (CC0 / public domain). See DATA_SOURCES.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as a plain script (python scripts/update_results.py) without installing the pkg.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import config, fixtures, live  # noqa: E402

# Schema version stamped into data/wc2026.json once the fixtures[] schedule block is present.
_SCHEMA_WITH_FIXTURES = 2

_FIXTURES_COMMENT = (
    "Structure-only match schedule (the offline base for the Matchday Home page): each entry is "
    "{team1, team2, group, round, venue, date, kickoff_utc}. Group matches use canonical team names "
    "(core.config.normalize_team); knockout matches use openfootball slot placeholders (e.g. '2A', "
    "'W74') until the teams are decided. kickoff_utc is the per-venue local time resolved to an "
    "absolute UTC instant. NO scores are committed here — live results are an optional overlay "
    "(known_results / the openfootball cache) merged on top by core.fixtures.resolve_fixtures. "
    "Regenerate with `python scripts/update_results.py --fixtures --write`. Source: "
    "openfootball/worldcup.json (CC0)."
)


def _bake_results(args: argparse.Namespace) -> None:
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
    print(
        f"\nWrote {len(results)} result(s) + as-of timestamp to {path.relative_to(config.BASE_DIR)}."
    )


def _bake_fixtures(args: argparse.Namespace) -> None:
    print(f"Fetching full schedule from {args.url or config.WC2026_LIVE_URL} ...")
    snapshot = fixtures.fetch_fixtures(url=args.url)
    schedule = fixtures.build_schedule(snapshot)
    print(f"  source:     {snapshot['source']}")
    print(f"  fetched_at: {snapshot['fetched_at']}")
    print(f"  {len(schedule)} fixture(s) (structure-only, no scores).")
    if "error" in snapshot:
        print(f"  (note: fetch degraded — {snapshot['error']})")

    if not schedule:
        print("No fixtures available; nothing to write.")
        return
    if not args.write:
        print(
            "\nDry run — data/wc2026.json unchanged. Re-run with --fixtures --write to bake these in."
        )
        return

    path = config.WC2026_PATH
    data = json.loads(path.read_text(encoding="utf-8"))
    data["schema_version"] = _SCHEMA_WITH_FIXTURES
    data["_comment_fixtures"] = _FIXTURES_COMMENT
    data["fixtures"] = schedule
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {len(schedule)} fixture(s) to {path.relative_to(config.BASE_DIR)}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh committed 2026 data from openfootball.")
    parser.add_argument("--url", default=None, help="override the openfootball feed URL")
    parser.add_argument(
        "--fixtures",
        action="store_true",
        help="operate on the fixtures[] schedule instead of known_results",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="actually write to data/wc2026.json (default: dry run, print only)",
    )
    args = parser.parse_args()

    if args.fixtures:
        _bake_fixtures(args)
    else:
        _bake_results(args)


if __name__ == "__main__":
    main()
