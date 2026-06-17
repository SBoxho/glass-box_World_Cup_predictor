"""Generate the committed current-squads snapshot ``data/squads2026.json`` from EA FC 26 ratings.

    python scripts/build_squads_snapshot.py [--force]

Downloads the public EA FC 26 player dataset (``config.SQUADS_2026_SOURCE_URL``; sofifa-derived
legacy schema, ``fifa_version == 26``) to the gitignored ``data/external/`` cache, keeps the top-N
(by overall) players for each of the 48 WC-2026 nations, and writes the small committed snapshot the
app + inference use. Only the derived JSON is committed — never the raw dump.

In-game ratings are **third-party estimates, not official data**.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import config, ingest  # noqa: E402

AS_OF = "2026-06-17"
# Attributes carried into the snapshot for the UI's per-attribute comparison (NaN → None, e.g. the
# six outfield aggregates are blank for goalkeepers). Squad-strength components use only `overall`.
ATTRS = ["pace", "shooting", "passing", "dribbling", "defending", "physic"]
SOURCE_NOTE = (
    "EA SPORTS FC 26 in-game player ratings (fifa_version=26), community sofifa-derived dataset via "
    "github.com/ismailoksuz/EAFC26-DataHub. Top-26 players by overall per WC-2026 nation. THIRD-PARTY "
    "ESTIMATES, not official data."
)
DISCLAIMER = "In-game ratings are third-party estimates, not official measurements."


def _download(force: bool) -> Path:
    dest = config.SQUADS_2026_RAW_CACHE_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        return dest
    req = urllib.request.Request(
        config.SQUADS_2026_SOURCE_URL, headers={"User-Agent": "glass-box-wc/1.0"}
    )
    with urllib.request.urlopen(req, timeout=300) as resp:  # noqa: S310 (trusted https mirror)
        dest.write_bytes(resp.read())
    return dest


def _int_or_none(value) -> int | None:
    return int(value) if pd.notna(value) else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build data/squads2026.json from EA FC 26 ratings."
    )
    parser.add_argument("--force", action="store_true", help="re-download the FC26 ratings")
    args = parser.parse_args()

    teams = sorted({t for grp in ingest.load_wc2026()["groups"].values() for t in grp})
    df = pd.read_csv(_download(args.force), low_memory=False)
    df["team"] = df["nationality_name"].map(config.normalize_team)
    df = df[df["team"].isin(teams)].copy()

    squads: dict[str, list[dict]] = {}
    for team, grp in df.groupby("team"):
        grp = grp.sort_values("overall", ascending=False).head(config.SQUADS_PER_TEAM)
        players = []
        for _, r in grp.iterrows():
            name = r["long_name"] if pd.notna(r["long_name"]) else r["short_name"]
            players.append(
                {
                    "name": str(name),
                    "club": str(r["club_name"]) if pd.notna(r["club_name"]) else "",
                    "position": str(r["player_positions"]).split(",")[0].strip(),
                    "fc26_ovr": int(r["overall"]),
                    **{a: _int_or_none(r.get(a)) for a in ATTRS},
                }
            )
        squads[team] = players

    missing = [t for t in teams if t not in squads]
    if missing:
        raise SystemExit(f"Missing WC nations after normalize (add TEAM_ALIASES): {missing}")

    out = {
        "as_of": AS_OF,
        "game_version": "EA Sports FC 26",
        "source": SOURCE_NOTE,
        "disclaimer": DISCLAIMER,
        "squads": squads,
    }
    config.SQUADS_2026_PATH.write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    sizes = {t: len(p) for t, p in squads.items()}
    thin = {t: n for t, n in sorted(sizes.items()) if n < config.SQUADS_PER_TEAM}
    print(
        f"Wrote {config.SQUADS_2026_PATH.relative_to(config.BASE_DIR)} — {len(squads)}/48 nations."
    )
    print(f"  players/team: min={min(sizes.values())}, max={max(sizes.values())}")
    if thin:
        print(f"  thin nations (<{config.SQUADS_PER_TEAM}, EA under-represents minnows): {thin}")


if __name__ == "__main__":
    main()
