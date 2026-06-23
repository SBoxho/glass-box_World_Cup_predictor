"""Render article-ready figures from a simulate_full.py JSON report.

    python scripts/make_figures.py                         # reads reports/sim_1M.json
    python scripts/make_figures.py --in reports/x.json     # custom input

Writes PNG (300 dpi) + SVG into reports/figures/:
    champion_odds.{png,svg}   horizontal bar — title probability, top 16 teams
    group_stage.{png,svg}     12 panels — P(advance) per team, with the real current standings
    road_to_final.{png,svg}   the projected bracket QF -> Champion (modal teams + reach odds)
    surprises.{png,svg}       underdog deep runs + projected group-winner upsets

Pure presentation of an already-computed report; no simulation here. matplotlib only (no external
rasteriser needed), so it runs anywhere the project does.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import config, flags  # noqa: E402

# Palette (self-contained — these are article assets, not theme-dependent UI).
INK = "#0f172a"
MUTE = "#64748b"
FAINT = "#cbd5e1"
BLUE = "#2563eb"
SKY = "#38bdf8"
GREEN = "#16a34a"
GOLD = "#f59e0b"
ROSE = "#e11d48"
VIOLET = "#7c3aed"
BG = "#ffffff"

plt.rcParams.update(
    {
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
        "font.family": "DejaVu Sans",
        "axes.edgecolor": FAINT,
        "axes.titlecolor": INK,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": MUTE,
        "ytick.color": INK,
    }
)

OUT = config.BASE_DIR / "reports" / "figures"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _subtitle(d: dict) -> str:
    m = d["meta"]
    return (
        f"{m['n_sims']:,} Monte-Carlo simulations · {m['n_locked_group_results']} real group "
        f"results locked in (as of {str(m['live_fetched_at'])[:10]}) · seed {m['seed']}"
    )


def _save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote figures/{name}.png + .svg")


# --------------------------------------------------------------------------------------
# 1) Title odds
# --------------------------------------------------------------------------------------
def fig_champion_odds(d: dict, top: int = 16) -> None:
    rows = sorted(d["teams"], key=lambda r: r["p_champion"], reverse=True)[:top]
    names = [flags.short_name(r["team"]) for r in rows][::-1]
    vals = [r["p_champion"] * 100 for r in rows][::-1]
    pmax = max(vals)
    colors = [BLUE if v >= pmax * 0.55 else (SKY if v >= pmax * 0.2 else FAINT) for v in vals]

    fig, ax = plt.subplots(figsize=(9, 7.6))
    bars = ax.barh(names, vals, color=colors, height=0.74)
    for bar, v in zip(bars, vals, strict=True):
        ax.text(
            v + pmax * 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{v:.1f}%",
            va="center",
            ha="left",
            fontsize=10.5,
            color=INK,
            fontweight="bold",
        )
    ax.set_xlim(0, pmax * 1.12)
    ax.set_xlabel("Probability of winning the tournament (%)", fontsize=11)
    ax.set_title("Who wins World Cup 2026?", fontsize=18, fontweight="bold", loc="left", pad=22)
    ax.annotate(
        _subtitle(d),
        (0, 1.012),
        xycoords="axes fraction",
        fontsize=9.5,
        color=MUTE,
        annotation_clip=False,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=0)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=FAINT, lw=0.6)
    _save(fig, "champion_odds")


# --------------------------------------------------------------------------------------
# 2) Group stage — P(advance) per team, with the real current standings
# --------------------------------------------------------------------------------------
def fig_group_stage(d: dict) -> None:
    groups = d["groups"]
    standings = d.get("current_standings", {})
    order = sorted(groups)
    fig, axes = plt.subplots(3, 4, figsize=(15.5, 10.2))
    fig.suptitle(
        "Group stage — who reaches the knockouts?",
        fontsize=19,
        fontweight="bold",
        x=0.07,
        ha="left",
        y=0.995,
    )
    fig.text(
        0.07,
        0.965,
        _subtitle(d) + "  ·  green = projected to advance (top-2 or best-third)",
        fontsize=10,
        color=MUTE,
        ha="left",
    )

    pts_by_team = {s["team"]: s for g in standings for s in standings.get(g, [])}
    for ax, g in zip(axes.ravel(), order, strict=False):
        teams = sorted(groups[g]["teams"], key=lambda t: t["p_advance"])
        names = [flags.short_name(t["team"]) for t in teams]
        vals = [t["p_advance"] * 100 for t in teams]
        colors = [GREEN if v >= 50 else FAINT for v in vals]
        bars = ax.barh(names, vals, color=colors, height=0.66)
        for bar, t, v in zip(bars, teams, vals, strict=True):
            played = pts_by_team.get(t["team"], {}).get("played", 0)
            tag = f"{v:.0f}%"
            if played:
                tag += f"  ·  {pts_by_team[t['team']]['pts']}pt"
            ax.text(
                2,
                bar.get_y() + bar.get_height() / 2,
                tag,
                va="center",
                ha="left",
                fontsize=9,
                color=INK if v >= 50 else MUTE,
                fontweight="bold",
            )
        ax.set_xlim(0, 100)
        ax.set_title(f"Group {g}", fontsize=12.5, fontweight="bold", loc="left")
        ax.set_xticks([])
        ax.tick_params(length=0)
        ax.tick_params(axis="y", labelsize=10)
        ax.spines[["top", "right", "bottom"]].set_visible(False)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.91, bottom=0.02, hspace=0.42, wspace=0.55)
    _save(fig, "group_stage")


# --------------------------------------------------------------------------------------
# 3) Road to the Final — projected bracket QF -> Champion
# --------------------------------------------------------------------------------------
def _box(ax, x, y, team, p, *, w=2.9, h=0.64, champ=False, lead=False):
    face = GOLD if champ else ("#eff6ff" if lead else "#f8fafc")
    edge = GOLD if champ else (BLUE if lead else FAINT)
    ax.add_patch(
        FancyBboxPatch(
            (x, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.7 if (champ or lead) else 1.0,
            edgecolor=edge,
            facecolor=face,
            mutation_aspect=1,
        )
    )
    fs = 11 if champ else 10
    ax.text(
        x + 0.14,
        y,
        f"{flags.short_name(team)}",
        va="center",
        ha="left",
        fontsize=fs,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        x + w - 0.12,
        y,
        f"{p * 100:.0f}%",
        va="center",
        ha="right",
        fontsize=9.5,
        color="#7c2d12" if champ else MUTE,
        fontweight="bold",
    )


def _elbow(ax, x0, y0, x1, y1, color=FAINT, lw=1.2):
    xm = (x0 + x1) / 2
    ax.plot(
        [x0, xm, xm, x1], [y0, y0, y1, y1], color=color, lw=lw, zorder=0, solid_capstyle="round"
    )


def fig_bracket(d: dict) -> None:
    rounds = {r["key"]: r for r in d["bracket"]["rounds"]}
    qf = rounds["QF"]["matches"]  # 4 matches
    sf = rounds["SF"]["matches"]  # 2 matches
    final = rounds["Final"]["matches"][0]
    champ = d["bracket"]["champion"][0]
    champ_team = champ["team"]

    fig, ax = plt.subplots(figsize=(15, 9))
    ax.axis("off")
    ax.set_xlim(0, 23)
    ax.set_ylim(-0.7, 10.4)
    W = 2.9
    x_qf, x_sf, x_fn, x_ch = 0.3, 6.9, 13.4, 18.9
    DY, DY2 = 0.62, 0.95

    def slot(m, side):
        return m[side][0] if m.get(side) else {"team": "—", "p": 0.0}

    # QF: 4 matches, each two stacked boxes centred on the match line (well spaced, no clipping).
    qf_centers = [8.6, 5.9, 3.2, 0.5]
    for cy, m in zip(qf_centers, qf, strict=True):
        for side, dy in (("top", DY), ("bottom", -DY)):
            s = slot(m, side)
            _box(ax, x_qf, cy + dy, s["team"], s["p"], w=W, lead=(s["team"] == champ_team))

    # SF: match k participants = advancers of QF 2k (top) / 2k+1 (bottom).
    sf_centers = [(qf_centers[0] + qf_centers[1]) / 2, (qf_centers[2] + qf_centers[3]) / 2]
    for mi, (cy, m) in enumerate(zip(sf_centers, sf, strict=True)):
        for side, dy, feeder in (("top", DY2, 2 * mi), ("bottom", -DY2, 2 * mi + 1)):
            s = slot(m, side)
            _elbow(
                ax,
                x_qf + W,
                qf_centers[feeder],
                x_sf,
                cy + dy,
                color=GOLD if s["team"] == champ_team else FAINT,
                lw=1.7 if s["team"] == champ_team else 1.1,
            )
            _box(ax, x_sf, cy + dy, s["team"], s["p"], w=W, lead=(s["team"] == champ_team))

    # Final: 2 boxes (SF advancers).
    fn_center = sum(sf_centers) / 2
    for side, dy, feeder in (("top", DY2, 0), ("bottom", -DY2, 1)):
        s = slot(final, side)
        _elbow(
            ax,
            x_sf + W,
            sf_centers[feeder],
            x_fn,
            fn_center + dy,
            color=GOLD if s["team"] == champ_team else FAINT,
            lw=1.7 if s["team"] == champ_team else 1.1,
        )
        _box(ax, x_fn, fn_center + dy, s["team"], s["p"], w=W, lead=(s["team"] == champ_team))

    # Champion.
    _elbow(ax, x_fn + W, fn_center, x_ch, fn_center, color=GOLD, lw=1.7)
    _box(ax, x_ch, fn_center, champ_team, champ["p"], w=3.1, h=0.82, champ=True)
    ax.text(
        x_ch + 1.55,
        fn_center + 0.68,
        "★ CHAMPION",
        ha="center",
        fontsize=10.5,
        color=GOLD,
        fontweight="bold",
    )

    for x, w, lab in (
        (x_qf, W, "Quarter-finals"),
        (x_sf, W, "Semi-finals"),
        (x_fn, W, "Final"),
        (x_ch, 3.1, "Winner"),
    ):
        ax.text(x + w / 2, 9.95, lab, ha="center", fontsize=11.5, fontweight="bold", color=MUTE)

    fig.suptitle("Road to the Final", fontsize=20, fontweight="bold", x=0.07, ha="left", y=0.99)
    fig.text(0.07, 0.95, _subtitle(d), fontsize=9.5, color=MUTE, ha="left")
    fig.text(
        0.07,
        0.03,
        "Most-likely team in each knockout slot, with its probability of reaching "
        "that slot across all simulations. Gold traces the projected champion's road.",
        fontsize=9,
        color=MUTE,
        ha="left",
    )
    fig.subplots_adjust(left=0.02, right=0.99, top=0.92, bottom=0.06)
    _save(fig, "road_to_final")


# --------------------------------------------------------------------------------------
# 4) Surprises — underdog deep runs
# --------------------------------------------------------------------------------------
def fig_surprises(d: dict) -> None:
    ud = d["surprises"]["underdog_deep_runs"][:7]
    fig, ax = plt.subplots(figsize=(9.5, 6.6))
    names = [f"{flags.short_name(r['team'])}  (Elo #{r['elo_rank']})" for r in ud][::-1]
    qf = [r["p_QF"] * 100 for r in ud][::-1]
    sf = [r["p_SF"] * 100 for r in ud][::-1]
    y = range(len(names))
    ax.barh(y, qf, color=FAINT, height=0.6, label="reach quarter-finals")
    ax.barh(y, sf, color=VIOLET, height=0.6, label="reach semi-finals")
    for i, (q, s) in enumerate(zip(qf, sf, strict=True)):
        ax.text(q + 1, i, f"QF {q:.0f}%  ·  SF {s:.0f}%", va="center", fontsize=9.5, color=MUTE)
    ax.set_yticks(list(y))
    ax.set_yticklabels(names, fontsize=10.5)
    ax.set_xlabel("Probability (%)", fontsize=11)
    ax.set_title(
        "The overachievers — deep runs from outside the top 8",
        fontsize=16,
        fontweight="bold",
        loc="left",
        pad=22,
    )
    ax.annotate(
        _subtitle(d),
        (0, 1.012),
        xycoords="axes fraction",
        fontsize=9.5,
        color=MUTE,
        annotation_clip=False,
    )
    ax.legend(loc="lower right", frameon=False, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=0)
    ax.xaxis.grid(True, color=FAINT, lw=0.6)
    ax.set_axisbelow(True)
    _save(fig, "surprises")


def main() -> None:
    ap = argparse.ArgumentParser(description="Render article figures from a simulate_full JSON.")
    ap.add_argument(
        "--in", dest="inp", type=Path, default=config.BASE_DIR / "reports" / "sim_1M.json"
    )
    args = ap.parse_args()
    inp = args.inp if args.inp.is_absolute() else (config.BASE_DIR / args.inp)
    d = _load(inp)
    print(f"Rendering figures from {inp.name} ({d['meta']['n_sims']:,} sims) -> reports/figures/")
    fig_champion_odds(d)
    fig_group_stage(d)
    fig_bracket(d)
    fig_surprises(d)
    print("Done.")


if __name__ == "__main__":
    main()
