"""Headless tournament Monte-Carlo → one clean JSON of per-team, per-group and bracket stats.

    python scripts/simulate_full.py                       # 1,000,000 sims, live results, seed 42
    python scripts/simulate_full.py --sims 100000         # quick run
    python scripts/simulate_full.py --no-live             # pre-tournament (ignore played results)
    python scripts/simulate_full.py --out reports/x.json  # custom output path

Why a script (not the Streamlit slider): a 1,000,000-run forecast is a ~40-minute, memory-steady
batch job that belongs in a headless, *reproducible* (fixed-seed) process with a committed artifact —
not an interactive in-browser session that Streamlit Cloud would time out. The app's simulator is
reused verbatim, so the numbers are identical to the UI at the same seed/standings; this just runs it
at scale and serialises everything an article needs.

Output JSON (see ``build_report``): run metadata, the locked live results fed in, the current real
group standings, every team's group-finish + stage-reach + title probabilities, per-group qualifying
odds, the most-likely-final / champion distributions, and a surprises/underdog analysis. All
probabilities are Monte-Carlo estimates (1σ ≈ sqrt(p(1-p)/N) ≈ 0.05pp at N=1e6 for p≈0.3).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from core import config, ingest, live, model, ranking, simulate, squads  # noqa: E402
from core.features import build_inference_state  # noqa: E402

DEFAULT_OUT = config.BASE_DIR / "reports" / "sim_1M.json"


# --------------------------------------------------------------------------------------
# Setup — reuse the processed caches (built by scripts/build_dataset.py) when present
# --------------------------------------------------------------------------------------
def _read_or_build_matches() -> pd.DataFrame:
    if config.MATCHES_PATH.exists():
        return pd.read_parquet(config.MATCHES_PATH)
    return ingest.get_clean_matches()


def _read_rankings():
    if config.RANKINGS_PATH.exists():
        return pd.read_parquet(config.RANKINGS_PATH)
    return ranking.load_rankings()


def _read_squads():
    if config.SQUAD_STRENGTH_PATH.exists():
        return pd.read_parquet(config.SQUAD_STRENGTH_PATH)
    return squads.load_squad_strength(history=False, snapshot=True)


def build_simulator(use_live: bool, seed: int):
    """Build the production predictor + simulator, optionally locking live group results."""
    matches = _read_or_build_matches()
    rankings = _read_rankings()
    squad_strength = _read_squads()
    state = build_inference_state(matches, rankings, squad_strength)
    artifact = model.load_model()
    predictor = model.Predictor(artifact, state)

    wc = ingest.load_wc2026()
    locked, fetched_at, source = [], None, None
    if use_live:
        snap = live.fetch_live_results()
        locked = snap.get("known_results", [])
        fetched_at, source = snap.get("fetched_at"), snap.get("source")
        wc = live.merge_known_results(wc, locked)

    fifa_pos = ranking.positions_as_of(rankings)
    sim = simulate.TournamentSimulator(predictor, wc, seed=seed, fifa_rank=fifa_pos)
    meta = {
        "trained_through": artifact.trained_through,
        "history_as_of": str(state.as_of.date()),
        "locked_results": locked,
        "fetched_at": fetched_at,
        "source": source,
    }
    return sim, predictor, wc, fifa_pos, meta


# --------------------------------------------------------------------------------------
# Real current standings from the locked results (what was actually fed in)
# --------------------------------------------------------------------------------------
def current_standings(wc: dict, locked: list[dict]) -> dict[str, list[dict]]:
    """Real group tables from the played (locked) matches, ordered pts → GD → GF."""
    out: dict[str, list[dict]] = {}
    by_pair = {(r["home"], r["away"]): r for r in locked}
    for g, teams in wc["groups"].items():
        stat = {t: {"team": t, "pts": 0, "gd": 0, "gf": 0, "ga": 0, "played": 0} for t in teams}
        for (h, a), r in by_pair.items():
            if h in stat and a in stat:
                hs, as_ = int(r["home_score"]), int(r["away_score"])
                for t, gf, ga in ((h, hs, as_), (a, as_, hs)):
                    stat[t]["gf"] += gf
                    stat[t]["ga"] += ga
                    stat[t]["gd"] += gf - ga
                    stat[t]["played"] += 1
                stat[h]["pts"] += 3 if hs > as_ else (1 if hs == as_ else 0)
                stat[a]["pts"] += 3 if as_ > hs else (1 if hs == as_ else 0)
        ordered = sorted(stat.values(), key=lambda s: (s["pts"], s["gd"], s["gf"]), reverse=True)
        if any(s["played"] for s in ordered):
            out[g] = ordered
    return out


# --------------------------------------------------------------------------------------
# Report assembly
# --------------------------------------------------------------------------------------
def _rank_map(values: dict[str, float], *, reverse=True) -> dict[str, int]:
    """team -> 1-based rank by ``values`` (reverse=True → highest value is rank 1)."""
    ordered = sorted(values, key=lambda t: values[t], reverse=reverse)
    return {t: i + 1 for i, t in enumerate(ordered)}


def build_report(sim, predictor, wc, fifa_pos, meta, result, elapsed: float) -> dict:
    teams = [t for ts in wc["groups"].values() for t in ts]
    ratings = predictor.state.ratings
    fifa_pts = getattr(predictor.state, "fifa_points", {}) or {}
    elo_rank = _rank_map({t: ratings.get(t, config.ELO_BASE) for t in teams})

    tbl = result.table.set_index("team")
    gtbl = result.group_table.set_index("team")
    champ_rank = _rank_map({t: float(tbl.loc[t, "Champion"]) for t in teams})
    sf_rank = _rank_map({t: float(tbl.loc[t, "SF"]) for t in teams})

    # --- per-team rows ---------------------------------------------------------------------
    team_rows = []
    for t in teams:
        row = {
            "team": t,
            "group": gtbl.loc[t, "group"],
            "elo": round(float(ratings.get(t, config.ELO_BASE)), 1),
            "elo_rank": elo_rank[t],
            "fifa_points": round(float(fifa_pts[t]), 1) if t in fifa_pts else None,
            "fifa_rank": fifa_pos.get(t),
            "p_win_group": round(float(gtbl.loc[t, "p1"]), 5),
            "p_runner_up": round(float(gtbl.loc[t, "p2"]), 5),
            "p_third": round(float(gtbl.loc[t, "p3"]), 5),
            "p_fourth": round(float(gtbl.loc[t, "p4"]), 5),
            "p_advance": round(float(gtbl.loc[t, "p_advance"]), 5),
            "p_R16": round(float(tbl.loc[t, "R16"]), 5),
            "p_QF": round(float(tbl.loc[t, "QF"]), 5),
            "p_SF": round(float(tbl.loc[t, "SF"]), 5),
            "p_final": round(float(tbl.loc[t, "Final"]), 5),
            "p_champion": round(float(tbl.loc[t, "Champion"]), 5),
            "champion_rank": champ_rank[t],
            # seed (Elo) minus simulated semis rank: >0 ⇒ outrunning its seeding (overperformer).
            "overperformance": elo_rank[t] - sf_rank[t],
        }
        team_rows.append(row)
    team_rows.sort(key=lambda r: r["p_champion"], reverse=True)

    # --- per-group qualifying picture ------------------------------------------------------
    groups = {}
    for g, gteams in wc["groups"].items():
        rows = sorted(
            (next(r for r in team_rows if r["team"] == t) for t in gteams),
            key=lambda r: r["p_advance"],
            reverse=True,
        )
        elo_fav = min(gteams, key=lambda t: elo_rank[t])  # strongest by Elo seed
        proj_winner = max(gteams, key=lambda t: float(gtbl.loc[t, "p1"]))
        groups[g] = {
            "teams": [
                {
                    "team": r["team"],
                    "elo": r["elo"],
                    "elo_rank": r["elo_rank"],
                    "p_win_group": r["p_win_group"],
                    "p_runner_up": r["p_runner_up"],
                    "p_advance": r["p_advance"],
                }
                for r in rows
            ],
            "elo_favourite": elo_fav,
            "projected_winner": proj_winner,
            "winner_is_upset": proj_winner != elo_fav,
        }

    # --- final + champion distributions ----------------------------------------------------
    n = result.n_sims
    finals = sorted(
        (
            {"teams": list(pair), "p": round(cnt / n, 5)}
            for pair, cnt in getattr(sim, "_last_final_pairs", {}).items()
        ),
        key=lambda d: d["p"],
        reverse=True,
    )[:15]
    champions = [
        {"team": r["team"], "p_champion": r["p_champion"], "elo_rank": r["elo_rank"]}
        for r in team_rows[:15]
    ]

    # --- surprises / underdogs -------------------------------------------------------------
    # Underdog deep runs: teams outside the Elo top 8, ranked by their chance of reaching the semis.
    underdogs = sorted(
        (r for r in team_rows if r["elo_rank"] > 8 and r["p_SF"] > 0.01),
        key=lambda r: r["p_SF"],
        reverse=True,
    )[:8]
    overperformers = sorted(team_rows, key=lambda r: r["overperformance"], reverse=True)[:6]
    underperformers = sorted(team_rows, key=lambda r: r["overperformance"])[:6]
    group_upsets = [
        {"group": g, "projected_winner": d["projected_winner"], "elo_favourite": d["elo_favourite"]}
        for g, d in groups.items()
        if d["winner_is_upset"]
    ]
    # Favourites in trouble: a group's Elo-strongest side projected under 60% to even advance.
    favourites_in_trouble = []
    for g, gteams in wc["groups"].items():
        fav = min(gteams, key=lambda t: elo_rank[t])
        adv = float(gtbl.loc[fav, "p_advance"])
        if adv < 0.60:
            favourites_in_trouble.append(
                {"team": fav, "group": g, "elo_rank": elo_rank[fav], "p_advance": round(adv, 5)}
            )
    favourites_in_trouble.sort(key=lambda d: d["p_advance"])

    return {
        "meta": {
            "n_sims": n,
            "seed": meta.get("seed"),
            "model_trained_through": meta["trained_through"],
            "match_history_as_of": meta["history_as_of"],
            "live_source": meta["source"],
            "live_fetched_at": meta["fetched_at"],
            "n_locked_group_results": len(meta["locked_results"]),
            "elapsed_seconds": round(elapsed, 1),
            "note": (
                "Monte-Carlo estimates; 1σ ≈ sqrt(p(1-p)/N). Knockout draws resolve by relative "
                "strength (extra-time/penalties proxy). EA FC squad ratings are third-party estimates."
            ),
        },
        "locked_results": meta["locked_results"],
        "current_standings": current_standings(wc, meta["locked_results"]),
        "champion_ranking": champions,
        "most_likely_final": {
            "teams": list(result.most_likely_final),
            "p": round(result.most_likely_final_prob, 5),
        },
        "final_matchups": finals,
        # Per-slot bracket projection (modal team per knockout position + advance odds) — the
        # "Road to the Final" structure, straight from core.bracket.build_bracket.
        "bracket": result.bracket,
        "groups": groups,
        "teams": team_rows,
        "surprises": {
            "underdog_deep_runs": [
                {
                    "team": r["team"],
                    "elo_rank": r["elo_rank"],
                    "p_QF": r["p_QF"],
                    "p_SF": r["p_SF"],
                    "p_champion": r["p_champion"],
                }
                for r in underdogs
            ],
            "overperformers": [
                {
                    "team": r["team"],
                    "elo_rank": r["elo_rank"],
                    "sim_overperformance": r["overperformance"],
                    "p_SF": r["p_SF"],
                }
                for r in overperformers
            ],
            "underperformers": [
                {
                    "team": r["team"],
                    "elo_rank": r["elo_rank"],
                    "sim_overperformance": r["overperformance"],
                    "p_advance": r["p_advance"],
                }
                for r in underperformers
            ],
            "group_winner_upsets": group_upsets,
            "favourites_in_trouble": favourites_in_trouble,
        },
    }


def _print_summary(report: dict) -> None:
    m = report["meta"]
    print(f"\n=== {m['n_sims']:,} simulations · seed {m['seed']} · {m['elapsed_seconds']}s ===")
    print(
        f"Live: {m['n_locked_group_results']} locked group results "
        f"(as of {m['live_fetched_at']}); model trained through {m['model_trained_through']}."
    )
    f = report["most_likely_final"]
    print(f"\nMost-likely final: {f['teams'][0]} vs {f['teams'][1]}  ({f['p']:.1%})")
    print("\nTitle odds (top 10):")
    for r in report["champion_ranking"][:10]:
        print(f"  {r['team']:<16} {r['p_champion']:6.2%}  (Elo #{r['elo_rank']})")
    print("\nUnderdog deep runs (outside Elo top 8, by P(semis)):")
    for r in report["surprises"]["underdog_deep_runs"][:6]:
        print(
            f"  {r['team']:<16} SF {r['p_SF']:5.1%} · QF {r['p_QF']:5.1%} · title {r['p_champion']:4.1%}  (Elo #{r['elo_rank']})"
        )
    if report["surprises"]["group_winner_upsets"]:
        print("\nProjected group-winner upsets (not the Elo favourite):")
        for u in report["surprises"]["group_winner_upsets"]:
            print(f"  Group {u['group']}: {u['projected_winner']} over {u['elo_favourite']}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Full tournament Monte-Carlo to a clean JSON report.")
    ap.add_argument(
        "--sims", type=int, default=1_000_000, help="number of tournaments (default 1e6)"
    )
    ap.add_argument("--seed", type=int, default=config.SEED, help="RNG seed (reproducible)")
    ap.add_argument("--no-live", action="store_true", help="ignore played results (pre-tournament)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output JSON path")
    args = ap.parse_args()

    out = args.out if args.out.is_absolute() else (config.BASE_DIR / args.out)
    t0 = time.time()
    sim, predictor, wc, fifa_pos, meta = build_simulator(use_live=not args.no_live, seed=args.seed)
    meta["seed"] = args.seed
    print(f"Setup done in {time.time() - t0:.1f}s — running {args.sims:,} simulations …")

    t1 = time.time()
    result = sim.run(n_sims=args.sims, seed=args.seed)
    elapsed = time.time() - t1

    report = build_report(sim, predictor, wc, fifa_pos, meta, result, elapsed)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _print_summary(report)
    print(f"\nWrote {out} ({out.stat().st_size // 1024} KB).")


if __name__ == "__main__":
    main()
