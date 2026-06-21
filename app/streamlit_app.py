"""Glass-Box World Cup Predictor — Streamlit front end.

A thin presentation layer: all modelling lives in ``core``. The model is loaded once from the
committed artifact (never retrained here); match history is downloaded/cached to build the live
inference state. The app opens on a **Matchday Home** landing page (see ``components/home.py``);
a top-level nav switches to the data-science views: Match Predictor, Tournament Simulator, and
Under the Hood.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Make `core` (project root) and `components` (this file's dir) importable when Streamlit runs
# this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent))  # app/ -> `components`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root -> `core`

from components import bracket as bracket_view  # noqa: E402  (presentation component)
from components import home  # noqa: E402  (presentation component; imports from core)
from components import team_page as team_page_view  # noqa: E402  (presentation component)

from core import bracket as core_bracket  # noqa: E402
from core import (  # noqa: E402
    config,
    explain,
    fixtures,
    ingest,
    live,
    model,
    ranking,
    simulate,
    squads,
)
from core.features import build_inference_state  # noqa: E402
from core.model import predict_from_features  # noqa: E402

st.set_page_config(
    page_title="Glass-Box World Cup 2026 Predictor",
    page_icon="⚽",
    layout="wide",
)

CLASS_COLORS = {"home": "#2563eb", "draw": "#9ca3af", "away": "#ef4444"}
POS_COLOR = "#16a34a"
NEG_COLOR = "#dc2626"

# How often the live-data views silently re-pull the openfootball feed (seconds). This is the
# cadence of the auto-refresh heartbeat *and* the cache TTL of the fetchers, so each scheduled
# rerun lands on freshly-evictable data. The feed is not real-time, so 5 minutes is a polite,
# practical balance; the client-side countdown clocks keep ticking every second regardless.
LIVE_REFRESH_SECS = 300


# --------------------------------------------------------------------------------------
# Auto-refresh heartbeat
# --------------------------------------------------------------------------------------
@st.fragment(run_every=LIVE_REFRESH_SECS)
def _live_heartbeat() -> None:
    """Invisible heartbeat that re-pulls the live feed on a fixed cadence — no button needed.

    Streamlit reruns *only this fragment* every ``LIVE_REFRESH_SECS``. Its first execution runs
    inline with the normal top-to-bottom script pass and merely *primes* (it sets a flag and
    returns); every later timer-driven rerun bumps the shared ``live_nonce`` and reruns the whole
    app via ``st.rerun(scope="app")``, so the cached ``ttl`` fetchers re-pull and every view shows
    the latest results.

    ``main()`` resets ``_hb_primed`` to ``False`` at the start of every full run, so the inline
    execution always primes (never reruns) and only genuine timer ticks fire the app-wide refresh.
    That reset-then-prime handshake is what prevents an infinite rerun loop.
    """
    if not st.session_state.get("_hb_primed", False):
        st.session_state["_hb_primed"] = True
        return
    st.session_state["live_nonce"] = st.session_state.get("live_nonce", 0) + 1
    st.rerun(scope="app")


# --------------------------------------------------------------------------------------
# Cached resources (loaded once)
# --------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model …")
def load_artifact():
    return model.load_model()


def _load_rankings_safe():
    """Best-effort FIFA ranking load: committed parquet, else the public feed + 2026 snapshot.

    The ranking is an *optional* enrichment (the model still runs on Elo/form, with
    ``fifa_points_diff`` falling back to its neutral default). So any failure here — a feed outage,
    or a stale ``core.config`` served by Streamlit Cloud's hot-reload after a deploy — degrades to
    Elo-only rather than crashing the whole app with a redacted error. Returns ``None`` on failure.
    """
    try:
        if config.RANKINGS_PATH.exists():
            return pd.read_parquet(config.RANKINGS_PATH)
        return ranking.load_rankings()
    except Exception:  # optional enrichment — never let it take the whole app down
        return None


def _load_squads_safe():
    """Best-effort squad-strength table for inference: committed parquet, else the snapshot only.

    Like :func:`_load_rankings_safe`, the squad signal is an *optional* enrichment — the model still
    runs on Elo/form/ranking with the squad features falling back to their neutral default. So this
    never downloads the heavy historical ratings at runtime: it uses the processed cache when present
    (local dev) and otherwise the committed 2026 squads snapshot alone (enough for current-strength
    predictions). Returns ``None`` on failure rather than taking the whole app down.
    """
    try:
        if config.SQUAD_STRENGTH_PATH.exists():
            return pd.read_parquet(config.SQUAD_STRENGTH_PATH)
        return squads.load_squad_strength(history=False, snapshot=True)
    except Exception:  # optional enrichment — never let it take the whole app down
        return None


@st.cache_data(show_spinner=False)
def load_squad_rosters():
    """The committed 26-man squads keyed by normalized nation, for the squad UI (best-effort)."""
    try:
        return squads.current_squads()
    except Exception:
        return {}


@st.cache_resource(show_spinner="Loading match history & computing current team strength …")
def load_state():
    if config.MATCHES_PATH.exists():
        matches = pd.read_parquet(config.MATCHES_PATH)
    else:  # deployed: processed data is not committed — build it from the public source
        matches = ingest.get_clean_matches()
    return matches, build_inference_state(matches, _load_rankings_safe(), _load_squads_safe())


@st.cache_resource(show_spinner=False)
def load_explainer(_artifact):
    return explain.build_explainer(_artifact)


@st.cache_data(show_spinner=False)
def load_wc():
    return ingest.load_wc2026()


@st.cache_data(show_spinner=False)
def load_metrics():
    if config.METRICS_PATH.exists():
        return json.loads(config.METRICS_PATH.read_text(encoding="utf-8"))
    return None


@st.cache_resource(show_spinner=False)
def load_fifa_positions():
    """Team → 1-based FIFA ranking position, the official group/best-third tiebreaker. Best-effort.

    Reuses the same ranking load as the model features (:func:`_load_rankings_safe`) and derives
    positions from the latest snapshot. Returns ``{}`` on any failure so the simulator simply skips
    the FIFA-ranking criterion (Elo/form still drive everything) rather than taking the app down.
    """
    rankings = _load_rankings_safe()
    if rankings is None:
        return {}
    try:
        return ranking.positions_as_of(rankings)
    except Exception:  # optional tiebreaker enrichment — never let it take the whole app down
        return {}


@st.cache_resource(show_spinner="Preparing matchup probabilities …")
def get_simulator(_predictor, _wc, _fifa_rank, key: str):
    return simulate.TournamentSimulator(_predictor, _wc, fifa_rank=_fifa_rank or None)


@st.cache_data(ttl=LIVE_REFRESH_SECS, show_spinner=False)
def fetch_live(nonce: int):
    """Pull played group results from the live feed. ``nonce`` is the cache key — the auto-refresh
    heartbeat bumps it every ``LIVE_REFRESH_SECS`` to force a real fetch; between ticks the same
    nonce is reused, so a fetch only hits the network once per cadence (or once per first load).
    Silent (no spinner) because it now runs automatically rather than on a button press."""
    return live.fetch_live_results()


@st.cache_data(show_spinner=False)
def resolve_schedule():
    """The Matchday page's default, offline-safe schedule: the committed ``data/wc2026.json``
    ``fixtures[]`` plus any cached live results merged on top. No network, so the page always loads
    and can always decide what to show. The live feed is pulled only when the user hits Refresh."""
    return fixtures.resolve_snapshot()


@st.cache_data(show_spinner=False)
def _played_pairs():
    """Already-played ties (real team names) from the offline-resolved schedule, for the bracket.

    Keyed by the unordered team pair so the bracket can stamp a final score on any knockout card
    whose two most-likely teams have actually met. Group results never collide with knockout
    pairings (they are always cross-group), so it is safe to build this from every finished fixture.
    Offline and best-effort — a failure just means no card shows a played score.
    """
    try:
        snap = fixtures.resolve_snapshot()
    except Exception:
        return {}
    out: dict = {}
    for f in snap.get("fixtures", []):
        if f.get("finished") and f.get("team1_score") is not None:
            out[frozenset((f["team1"], f["team2"]))] = {
                "score": {f["team1"]: f["team1_score"], f["team2"]: f["team2_score"]}
            }
    return out


@st.cache_data(ttl=LIVE_REFRESH_SECS, show_spinner=False)
def fetch_fixtures(nonce: int):
    """Pull the full live 2026 schedule (kickoffs, venues, scores) for the Matchday page.

    Cached for ``LIVE_REFRESH_SECS`` (the auto-refresh heartbeat bumps ``nonce`` to force a fresh
    pull each cadence). Degrades to the openfootball cache (or an empty snapshot) offline; the
    caller falls back to the committed schedule if it comes back empty, so the page always renders.
    Silent (no spinner) because it now runs automatically rather than on a button press."""
    return fixtures.fetch_fixtures()


# --------------------------------------------------------------------------------------
# Match prediction helpers
# --------------------------------------------------------------------------------------
def predict_ui(predictor, artifact, explainer, team_a, team_b):
    """Return (display_probs, explanation) for team_a vs team_b at a neutral venue.

    Every World Cup match is treated as neutral here (the format is effectively neutral-site, and
    even host games are not consistently home), so there is no venue control. Neutral predictions
    are mirror-symmetrized inside :func:`predict_from_features`, so the orientation is exact and the
    home (``H``) / away (``A``) classes map directly onto team_a / team_b.
    """
    feats = predictor.features(team_a, team_b, neutral=True, is_host_home=0)
    probs = predict_from_features(artifact, feats, neutral=True)
    display = {team_a: probs["H"], "Draw": probs["D"], team_b: probs["A"]}
    ex = explain.explain_match(artifact, feats, probs, team_a, team_b, explainer)
    return display, ex


def probability_bar(display: dict, team_a: str, team_b: str) -> go.Figure:
    labels = [team_a, "Draw", team_b]
    vals = [display[team_a], display["Draw"], display[team_b]]
    colors = [CLASS_COLORS["home"], CLASS_COLORS["draw"], CLASS_COLORS["away"]]
    fig = go.Figure(
        go.Bar(
            x=vals,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.0%}" for v in vals],
            textposition="auto",
        )
    )
    fig.update_layout(
        height=200,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(range=[0, 1], tickformat=".0%", title=None),
        yaxis=dict(autorange="reversed"),
        showlegend=False,
    )
    return fig


def shap_bar(ex: dict, top_n: int = 8) -> go.Figure:
    contribs = ex["contributions"][:top_n][::-1]
    vals = [c["shap"] for c in contribs]
    labels = [c["label"] for c in contribs]
    colors = [POS_COLOR if v >= 0 else NEG_COLOR for v in vals]
    fig = go.Figure(
        go.Bar(
            x=vals,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.2f}" for v in vals],
            textposition="auto",
        )
    )
    fig.update_layout(
        height=max(240, 38 * len(contribs)),
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title=f"SHAP contribution toward {ex['favored']}",
        showlegend=False,
    )
    return fig


# --------------------------------------------------------------------------------------
# Squad panel (EA FC 26 ratings — third-party estimates)
# --------------------------------------------------------------------------------------
_ATTR_KEYS = ["pace", "shooting", "passing", "dribbling", "defending", "physic"]
_ATTR_LABELS = ["Pace", "Shooting", "Passing", "Dribbling", "Defending", "Physical"]


def _mean(vals: list) -> float | None:
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def squad_metrics(roster: list[dict]) -> dict:
    """Summarise a 26-man roster (sorted by OVR, desc) for the squad panel."""
    ovrs = [p["fc26_ovr"] for p in roster]
    best_xi = _mean(ovrs[:11])
    depth = _mean(ovrs[11:26]) if len(ovrs) > 11 else best_xi
    # Attribute averages over the best XI's outfield players (GKs carry blank outfield attributes).
    attrs = {a: _mean([p.get(a) for p in roster[:11]]) for a in _ATTR_KEYS}
    return {
        "best_xi": best_xi,
        "star3": _mean(ovrs[:3]),
        "depth": depth,
        "attrs": attrs,
        "n": len(roster),
    }


def squad_radar(name_a: str, attrs_a: dict, name_b: str, attrs_b: dict) -> go.Figure:
    """Radar comparing the two squads' best-XI attribute averages (pace/shooting/… )."""
    fig = go.Figure()
    for name, attrs, color in (
        (name_a, attrs_a, CLASS_COLORS["home"]),
        (name_b, attrs_b, CLASS_COLORS["away"]),
    ):
        vals = [attrs.get(k) or 0 for k in _ATTR_KEYS]
        fig.add_trace(
            go.Scatterpolar(
                r=[*vals, vals[0]],
                theta=[*_ATTR_LABELS, _ATTR_LABELS[0]],
                fill="toself",
                name=name,
                line_color=color,
                opacity=0.6,
            )
        )
    fig.update_layout(
        height=320,
        margin=dict(l=30, r=30, t=30, b=30),
        polar=dict(radialaxis=dict(range=[40, 92], showticklabels=True, tickfont=dict(size=9))),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, x=0),
    )
    return fig


def _top_players_df(roster: list[dict], n: int = 8) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Player": p["name"],
                "Pos": p["position"],
                "Club": p.get("club", ""),
                "OVR": p["fc26_ovr"],
            }
            for p in roster[:n]
        ]
    )


def render_squad_panel(team_a: str, team_b: str, rosters: dict) -> None:
    """Render the EA FC 26 squad comparison: per-team strength + top players + an attribute radar."""
    st.markdown("**Squads — EA Sports FC 26**")
    ra, rb = rosters.get(team_a), rosters.get(team_b)
    if not ra or not rb:
        missing = team_a if not ra else team_b
        st.caption(
            f"Squad data unavailable for {missing} — the model falls back to a neutral default."
        )
        return

    ma, mb = squad_metrics(ra), squad_metrics(rb)
    cols = st.columns(2)
    for col, team, m, roster in ((cols[0], team_a, ma, ra), (cols[1], team_b, mb, rb)):
        with col:
            st.markdown(f"**{team}**")
            x1, x2, x3 = st.columns(3)
            x1.metric("Best XI OVR", f"{m['best_xi']:.1f}")
            x2.metric("Top-3 stars", f"{m['star3']:.1f}")
            x3.metric("Depth (12–26)", f"{m['depth']:.1f}" if m["depth"] else "—")
            st.dataframe(_top_players_df(roster), hide_index=True, width="stretch", height=180)
            if m["n"] < config.SQUADS_PER_TEAM:
                st.caption(f"Only {m['n']} rated players (EA under-represents this nation).")

    st.plotly_chart(squad_radar(team_a, ma["attrs"], team_b, mb["attrs"]), width="stretch")
    st.caption(
        "EA Sports FC 26 in-game ratings — **third-party estimates, not official data**. Best-XI = "
        "mean overall of the top 11; attributes averaged over the best XI. Fed to the model as "
        "point-in-time squad features (see *Under the Hood* for the with-vs-without ablation)."
    )


# --------------------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------------------
def tab_predictor(predictor, artifact, explainer, wc, state, rosters):
    st.subheader("Match Predictor")
    st.caption(
        "Pick two teams. The model returns calibrated win / draw / loss probabilities at a neutral "
        "venue (the World Cup format), the SHAP contributions behind them, and a plain-English read."
    )

    teams = sorted({t for grp in wc["groups"].values() for t in grp})
    c1, c2 = st.columns(2)
    with c1:
        team_a = st.selectbox("Team A", teams, index=teams.index("Brazil"))
    with c2:
        team_b = st.selectbox("Team B", teams, index=teams.index("France"))

    if team_a == team_b:
        st.warning("Pick two different teams.")
        return

    display, ex = predict_ui(predictor, artifact, explainer, team_a, team_b)

    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Outcome probabilities**")
        st.plotly_chart(probability_bar(display, team_a, team_b), width="stretch")
        elo_a = state.ratings.get(team_a, config.ELO_BASE)
        elo_b = state.ratings.get(team_b, config.ELO_BASE)
        m1, m2 = st.columns(2)
        m1.metric(f"{team_a} Elo", f"{elo_a:.0f}")
        m2.metric(f"{team_b} Elo", f"{elo_b:.0f}", delta=f"{elo_a - elo_b:+.0f} vs {team_a}")
        fifa_a, fifa_b = state.fifa_points.get(team_a), state.fifa_points.get(team_b)
        if fifa_a is not None and fifa_b is not None:
            g1, g2 = st.columns(2)
            g1.metric(f"{team_a} FIFA pts", f"{fifa_a:.0f}")
            g2.metric(
                f"{team_b} FIFA pts", f"{fifa_b:.0f}", delta=f"{fifa_a - fifa_b:+.0f} vs {team_a}"
            )
        st.info(ex["narrative"])
    with right:
        st.markdown("**Why? — SHAP feature contributions**")
        st.plotly_chart(shap_bar(ex), width="stretch")
        st.caption(
            "Green pushes toward the favoured side; red pushes against. SHAP runs on the raw "
            "tree model — calibration only rescales the final probabilities, not these signs."
        )

    if rosters:
        st.divider()
        render_squad_panel(team_a, team_b, rosters)


def _results_fingerprint(locked: list[dict]) -> str:
    """Stable content hash of the locked group results.

    Used as the standings part of the simulator cache key and to decide when a cached simulation
    is stale. It folds in *only* the scores (home/away teams + goals), not the fetch timestamp, so
    an auto-refresh that re-pulls identical results does **not** needlessly rebuild the simulator or
    discard the user's current run — those only change when a scoreline actually changes.
    """
    payload = sorted((r["home"], r["away"], r["home_score"], r["away_score"]) for r in locked)
    return hashlib.md5(json.dumps(payload).encode()).hexdigest()  # noqa: S324 (non-crypto cache key)


def _effective_results(wc: dict, snapshot: dict | None):
    """Merge committed + live ``known_results``; return (wc_eff, locked, fetched_at, source).

    A live snapshot (from the Refresh button) takes precedence; otherwise any results committed in
    wc2026.json are used; otherwise the simulator runs from a blank pre-tournament state.
    """
    committed = wc.get("known_results", [])
    live_results = snapshot.get("known_results", []) if snapshot else []
    if live_results:
        wc_eff = live.merge_known_results(wc, live_results)
        return wc_eff, wc_eff["known_results"], snapshot.get("fetched_at"), snapshot.get("source")
    if committed:
        return wc, committed, wc.get("known_results_as_of"), "data/wc2026.json (committed)"
    return wc, [], None, None


def _render_live_status(snapshot, locked, fetched_at, source) -> None:
    """Show the 'as of' line + a locked-results expander, or an honest fallback message."""
    if snapshot is not None and snapshot.get("error") and not locked:
        st.warning(
            "Couldn't fetch live results (offline or source unavailable). "
            "Simulating from the pre-tournament state."
        )
        return
    if not locked:
        if snapshot is not None:
            st.info(
                "Live feed reached, but no played group matches were found yet. "
                "Simulating from the pre-tournament state."
            )
        else:
            st.caption(
                "No live results loaded — simulating from the pre-tournament state. "
                "Click **🔄 Refresh live results** to pull current standings."
            )
        return
    st.caption(
        f"📡 **{len(locked)}** group match(es) locked · as of {fetched_at} · source: {source}"
    )
    with st.expander(f"Locked group results ({len(locked)})", expanded=False):
        rows = [
            {
                "Home": r["home"],
                "Score": f"{r['home_score']}–{r['away_score']}",
                "Away": r["away"],
            }
            for r in locked
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def tab_simulator(predictor, artifact, wc):
    st.subheader("Tournament Simulator")
    st.caption(
        "Monte Carlo over the full 48-team bracket: 12 groups → top 2 + 8 best thirds → "
        "Round of 32 → Final. Each match is sampled from the model's calibrated probabilities."
    )

    # --- Live results: lock already-played group matches so the sim runs forward ---------------
    # Pulled automatically (cached, refreshed by the heartbeat) — no button. The committed/empty
    # fallbacks inside the fetcher keep this offline-safe.
    nonce = st.session_state.get("live_nonce", 0)
    snapshot = fetch_live(nonce)
    wc_eff, locked, fetched_at, source = _effective_results(wc, snapshot)
    _render_live_status(snapshot, locked, fetched_at, source)

    # Drop a cached run only when the standings actually change (not on every silent refresh), so an
    # auto-refresh that re-pulls identical results leaves the user's current simulation in place.
    fingerprint = _results_fingerprint(locked)
    if st.session_state.get("sim_fingerprint") != fingerprint:
        st.session_state.pop("sim_result", None)
        st.session_state["sim_fingerprint"] = fingerprint

    # --- Simulation controls -------------------------------------------------------------------
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        n_sims = st.select_slider("Simulations", options=[2000, 5000, 10000, 20000], value=10000)
    with c2:
        seed = st.number_input("Seed", value=config.SEED, step=1)
    with c3:
        st.write("")
        st.write("")
        run = st.button("Run simulation", type="primary", width="stretch")

    fifa_rank = load_fifa_positions()
    sim = get_simulator(
        predictor,
        wc_eff,
        fifa_rank,
        f"{artifact.trained_through}|{fingerprint}|fifa{len(fifa_rank)}",
    )

    if run:
        with st.spinner(f"Simulating {n_sims:,} tournaments …"):
            st.session_state["sim_result"] = sim.run(n_sims=int(n_sims), seed=int(seed))

    result = st.session_state.get("sim_result")
    if result is None:
        st.info("Set the options and hit **Run simulation**.")
        return

    fin = result.most_likely_final
    st.success(
        f"**Most likely final:** {fin[0]} vs {fin[1]}  "
        f"({result.most_likely_final_prob:.1%} of {result.n_sims:,} simulations)"
    )

    # --- Knockout bracket: the visual, story-led view of the same simulation ---------------------
    ratings = getattr(predictor.state, "ratings", None)
    highlights = core_bracket.derive_highlights(result, ratings)

    st.markdown("### 🗺️ Road to the Final")
    st.caption(
        "Each card is a projected knockout tie from the Monte-Carlo runs: the teams most likely to "
        "fill each slot (with their chance to get there), and the simulated winner split. The four "
        "highlights below trace stories through the bracket."
    )
    bracket_view.render_highlights(highlights)

    teams_sorted = result.table["team"].tolist()
    csel, cleg = st.columns([1, 2])
    with csel:
        sel = st.selectbox("Trace a team's path", ["— none —", *teams_sorted], key="bracket_team")
    selected_team = None if sel.startswith("—") else sel
    with cleg:
        st.write("")
        bracket_view.render_legend(selected_team)

    st.caption("↔ Scroll horizontally to follow the rounds. Percentages come from the same runs.")
    bracket_view.render_bracket(
        result,
        ratings,
        selected_team=selected_team,
        played=_played_pairs(),
        highlights=highlights,
    )

    # --- The original table + heatmap, kept available as a data-grounded backstop ----------------
    with st.expander("📋 Raw probabilities — full table & heatmap", expanded=False):
        table = result.table.copy()
        pct_cols = simulate.STAGES
        st.markdown("**Title odds & run probabilities**")
        st.dataframe(
            table.rename(columns={"team": "Team", "group": "Grp"}),
            width="stretch",
            hide_index=True,
            column_config={
                # ProgressColumn fills the bar over [min,max] and "percent" renders the value
                # ×100 with a % sign — so a 0.99 reach-probability shows as "99.00%", not "1.0%".
                c: st.column_config.ProgressColumn(
                    c, format="percent", min_value=0.0, max_value=1.0
                )
                for c in pct_cols
            },
            height=460,
        )
        st.markdown("**How far each team goes** (top 16 by title probability)")
        st.plotly_chart(stage_heatmap(table.head(16)), width="stretch")


def _team_sim_result(sim, key: str, n_sims: int, seed: int):
    """Cache a default simulation for the team page in session state (keyed by standings + config).

    The team page needs a :class:`SimulationResult` to slice. If the user has already run the
    Tournament Simulator we reuse that run (see ``tab_my_team``); otherwise we auto-run a default
    once and reuse it across team switches and reruns, so picking a team feels instant. The key
    folds in the locked-results count + fetch time, so a live refresh invalidates it.
    """
    cache = st.session_state.setdefault("_team_sim_cache", {})
    ck = (key, n_sims, seed)
    if ck not in cache:
        with st.spinner(f"Simulating {n_sims:,} tournaments for your team page …"):
            cache[ck] = sim.run(n_sims=n_sims, seed=seed)
    return cache[ck]


def tab_my_team(predictor, artifact, explainer, wc, state):
    """Choose-Your-Team route: resolve standings + a simulation, then hand off to the component.

    Reuses the same live-results merge and cached simulator as the Tournament Simulator, so the
    team page reflects current standings. Prefers an existing full run if the user already
    simulated; otherwise auto-runs a default once (cached).
    """
    nonce = st.session_state.get("live_nonce", 0)
    snapshot = fetch_live(nonce)
    wc_eff, locked, fetched_at, source = _effective_results(wc, snapshot)
    fifa_rank = load_fifa_positions()
    key = f"{artifact.trained_through}|{_results_fingerprint(locked)}|fifa{len(fifa_rank)}"
    sim = get_simulator(predictor, wc_eff, fifa_rank, key)
    result = st.session_state.get("sim_result")
    if result is None:
        result = _team_sim_result(sim, key, 10000, int(config.SEED))

    schedule = resolve_schedule()
    tz, _ = home.resolve_timezone()
    team_page_view.render_team_page(
        predictor,
        artifact,
        explainer,
        wc,
        state,
        result,
        fixtures_list=(schedule or {}).get("fixtures") or [],
        played=_played_pairs(),
        tz=tz,
        known_results=locked,
        source=source,
        fetched_at=fetched_at,
    )


def stage_heatmap(table: pd.DataFrame) -> go.Figure:
    stages = simulate.STAGES
    z = table[stages].values
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=stages,
            y=table["team"],
            colorscale="Blues",
            zmin=0,
            zmax=1,
            text=[[f"{v:.0%}" for v in row] for row in z],
            texttemplate="%{text}",
            colorbar=dict(title="P", tickformat=".0%"),
        )
    )
    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def tab_under_the_hood(metrics, wc, artifact):
    st.subheader("Under the Hood")
    st.caption("How the model is built, how well it is calibrated, and where it falls short.")

    st.markdown(
        "**Pipeline:** international results → point-in-time Elo & form features (no leakage) "
        "→ calibrated XGBoost (H/D/A) → SHAP explanations → Monte Carlo tournament simulation."
    )

    if metrics is None:
        st.warning("metrics.json not found — run `python scripts/train.py`.")
    else:
        m, b = metrics["model"], metrics["baselines"]
        sp = metrics["split"]
        st.markdown(
            f"**Temporal backtest** — trained on {sp['trainval'][0]} → {sp['trainval'][1]} "
            f"({sp['trainval'][2]:,} matches), tested on the most recent {sp['test'][2]:,} "
            f"({sp['test'][0]} → {sp['test'][1]}). No random folds — strictly chronological."
        )
        rows = [
            {
                "Model": "Calibrated XGBoost",
                "Log-loss ↓": m["logloss"],
                "Accuracy ↑": m["accuracy"],
                "ECE ↓": m["reliability"]["ece"],
            },
            {
                "Model": "Baseline: Elo-only",
                "Log-loss ↓": b["elo_only"]["logloss"],
                "Accuracy ↑": b["elo_only"]["accuracy"],
                "ECE ↓": float("nan"),
            },
        ]
        if "fifa_only" in b:
            rows.append(
                {
                    "Model": "Baseline: FIFA-ranking-only",
                    "Log-loss ↓": b["fifa_only"]["logloss"],
                    "Accuracy ↑": b["fifa_only"]["accuracy"],
                    "ECE ↓": float("nan"),
                }
            )
        if "squad_only" in b:
            rows.append(
                {
                    "Model": "Baseline: squad-only (EA FC)",
                    "Log-loss ↓": b["squad_only"]["logloss"],
                    "Accuracy ↑": b["squad_only"]["accuracy"],
                    "ECE ↓": float("nan"),
                }
            )
        rows.append(
            {
                "Model": "Baseline: always-home",
                "Log-loss ↓": b["always_home"]["logloss"],
                "Accuracy ↑": b["always_home"]["accuracy"],
                "ECE ↓": float("nan"),
            }
        )
        comp = pd.DataFrame(rows)
        st.dataframe(
            comp,
            hide_index=True,
            width="stretch",
            column_config={
                "Log-loss ↓": st.column_config.NumberColumn(format="%.4f"),
                "Accuracy ↑": st.column_config.NumberColumn(format="%.3f"),
                "ECE ↓": st.column_config.NumberColumn(format="%.4f"),
            },
        )
        st.caption(
            "Honest result: against a strong Elo baseline the ML lift is small — the model "
            "essentially matches Elo on accuracy and log-loss. Its value is **calibration** "
            "(near-zero ECE), **explainability** (SHAP), and **simulation**, not raw accuracy."
        )
        r = (metrics.get("feature_notes") or {}).get("elo_fifa_pearson")
        if r is not None:
            st.caption(
                f"**FIFA ranking vs Elo:** the FIFA-points gap and the Elo gap are strongly "
                f"correlated (Pearson r ≈ {r:.2f}) — they measure overlapping strength. On this "
                f"backtest the FIFA-only baseline is in fact a bit *weaker* than Elo-only, and "
                f"adding FIFA points on top of Elo gives no measurable lift (the model still just "
                f"ties the Elo baseline). It is included for transparency and as a baseline, not "
                f"because it moves the model — the honest takeaway is that Elo already captures it."
            )

        ablation = metrics.get("ablation")
        if ablation:
            st.markdown(
                "**Squad-strength ablation** — the model with vs without the EA FC squad features"
            )
            aw, ao = ablation["with_squad"], ablation["without_squad"]
            ab_df = pd.DataFrame(
                [
                    {
                        "Model": "With squad features",
                        "Log-loss ↓": aw["logloss"],
                        "Accuracy ↑": aw["accuracy"],
                        "ECE ↓": aw["ece"],
                    },
                    {
                        "Model": "Without squad features",
                        "Log-loss ↓": ao["logloss"],
                        "Accuracy ↑": ao["accuracy"],
                        "ECE ↓": ao["ece"],
                    },
                ]
            )
            st.dataframe(
                ab_df,
                hide_index=True,
                width="stretch",
                column_config={
                    "Log-loss ↓": st.column_config.NumberColumn(format="%.4f"),
                    "Accuracy ↑": st.column_config.NumberColumn(format="%.3f"),
                    "ECE ↓": st.column_config.NumberColumn(format="%.4f"),
                },
            )
            rsq = (metrics.get("feature_notes") or {}).get("elo_squad_pearson")
            rsq_txt = (
                f" Squad strength is moderately collinear with Elo (r ≈ {rsq:.2f})." if rsq else ""
            )
            st.caption(
                f"Honest result: adding the four EA FC squad features (squad strength, attack-vs-"
                f"defence, depth, star power) changes the backtest by only **{ablation['delta_logloss']:+.4f}** "
                f"log-loss and **{ablation['delta_accuracy']:+.3f}** accuracy — a tiny, expected lift."
                f"{rsq_txt} The model still essentially ties the strong Elo-only baseline. In-game "
                f"ratings are **third-party estimates**, included for explainability and transparency, "
                f"not because they move the model."
            )

    c1, c2 = st.columns(2)
    with c1:
        if config.RELIABILITY_PLOT_PATH.exists():
            st.image(str(config.RELIABILITY_PLOT_PATH), caption="Reliability curve (test set)")
    with c2:
        if config.SHAP_SUMMARY_PATH.exists():
            st.image(str(config.SHAP_SUMMARY_PATH), caption="Global SHAP feature importance")

    with st.expander("Methodology & limitations", expanded=False):
        st.markdown(
            """
- **Data:** `martj42/international_results` (every men's international since 1872). Trained on
  matches from 2002 onward; full history warms up Elo.
- **Elo:** chess-style, K scaled by goal margin and tournament importance; home advantage in
  the expectation, zeroed at neutral venues.
- **Features (point-in-time, no leakage):** Elo gap, FIFA-ranking gap, recent form & goal
  difference, decayed head-to-head, venue, host advantage, days of rest, and EA FC squad
  strength (best-XI / attack-vs-defence / depth / star power — third-party in-game estimates,
  attached from the ratings version current at each match date).
- **Model:** multiclass XGBoost, isotonic-calibrated via cross-validation so the numbers are
  usable as real probabilities. Neutral-venue predictions are symmetrized.
- **Simulation:** outcomes sampled from calibrated probabilities; scorelines from Poisson
  draws for tiebreakers; knockout draws resolved by relative strength (extra-time proxy). The
  knockout phase follows the **official 2026 rules** — the fixed match DAG (M73→M104) and FIFA's
  495-row **Annexe C** third-place matrix, loaded from a committed rules file as the single source
  of truth — and group ordering uses the official tiebreakers (head-to-head → overall goal
  difference / goals → FIFA ranking).

**Limitations** — international football is a small-data, high-variance sport: upsets are
common and a single tournament is one noisy draw from these distributions. Squad changes,
injuries, and form swings are only partially captured. Two group tiebreaker steps are not
modelled and fall back to a deterministic random draw: the **team-conduct (fair-play) score** (no
cards are simulated) and the **"successively earlier FIFA rankings"** step (only the latest
snapshot is on file). **This is an educational / decision-support tool — probabilistic, not betting
advice.**
            """
        )


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    artifact = load_artifact()
    matches, state = load_state()
    explainer = load_explainer(artifact)
    wc = load_wc()
    metrics = load_metrics()
    rosters = load_squad_rosters()
    predictor = model.Predictor(artifact, state)

    # Slim brand line + top-level navigation. Matchday Home is the default landing view, so the
    # app opens on "what's happening now" rather than on the model dashboard.
    st.markdown("##### ⚽ Glass-Box World Cup 2026 Predictor")
    st.session_state.setdefault(home.NAV_KEY, home.VIEW_MATCHDAY)
    views = [
        home.VIEW_MATCHDAY,
        home.VIEW_PREDICT,
        home.VIEW_SIMULATE,
        home.VIEW_TEAM,
        home.VIEW_BRACKET,
    ]
    nav = st.segmented_control("Navigation", views, key=home.NAV_KEY, label_visibility="collapsed")
    view = nav or home.VIEW_MATCHDAY  # single-select can be cleared to None → fall back to Home

    # Auto-refresh: re-prime the heartbeat on every full run, then mount it only on the live-data
    # views (Matchday / Simulate / My Team) so Predict / Under-the-Hood don't rerun on the timer.
    st.session_state["_hb_primed"] = False
    if view in (home.VIEW_MATCHDAY, home.VIEW_SIMULATE, home.VIEW_TEAM):
        _live_heartbeat()

    if view == home.VIEW_MATCHDAY:
        # Always overlay the live schedule (auto-refreshed by the heartbeat), falling back to the
        # committed offline schedule whenever the feed is empty or unreachable, so the page never
        # loses its floor. The shared ``live_nonce`` keys the cached fetch.
        nonce = st.session_state.get("live_nonce", 0)
        snapshot = resolve_schedule()
        live_snapshot = fetch_fixtures(nonce)
        if live_snapshot.get("fixtures"):
            snapshot = live_snapshot
        home.render_home(predictor, artifact, explainer, wc, state, snapshot)
    else:
        st.caption(
            f"Model trained through **{artifact.trained_through}** · "
            f"match history to **{state.as_of.date()}** · {len(state.teams)} national teams."
        )
        if view == home.VIEW_PREDICT:
            tab_predictor(predictor, artifact, explainer, wc, state, rosters)
        elif view == home.VIEW_SIMULATE:
            tab_simulator(predictor, artifact, wc)
        elif view == home.VIEW_TEAM:
            tab_my_team(predictor, artifact, explainer, wc, state)
        else:
            tab_under_the_hood(metrics, wc, artifact)

    st.divider()
    st.caption(
        "Educational / decision-support tool. Outputs are **probabilistic**, not guarantees, "
        "and this is **not betting advice**. Model trained on historical international results; "
        "upsets happen. Made with calibrated XGBoost + SHAP + Monte Carlo."
    )


# Streamlit re-runs this script top-to-bottom on every interaction, so just call main().
main()
