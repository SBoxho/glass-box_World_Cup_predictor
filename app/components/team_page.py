"""Choose Your Team — a fan-facing team page built on the simulator's own outputs.

Pick one of the 48 finalists and the page personalises into a "follow your team" view: a hero with
the team's title odds (framed against the field favourite so the uncertainty is visible), a
stage-by-stage run funnel, the group outlook (model advance odds + live standings), the next-match
prediction, a field-relative strengths-&-weaknesses read, the team's road traced on the bracket, and
a plain-English "what needs to happen?".

Pure presentation (the ``components`` rule): every number comes from :mod:`core.team_page` (which
slices an already-computed :class:`core.simulate.SimulationResult` + the inference state) or from a
single model call for the next fixture — this module only lays it out. The selected team is held in
``st.session_state`` so it persists across reruns and tab switches. Nothing here invents tactical
claims: a factor with no data (squad strength on a build without the EA FC feature; FIFA points if
the feed is down) renders an honest "limited coverage" note rather than a guessed rating.
"""

from __future__ import annotations

import html
from datetime import UTC, datetime

import plotly.graph_objects as go
import streamlit as st

from components import bracket as bracket_view
from components import home as home_view
from core import explain
from core import team_page as tp
from core.model import predict_from_features

SESSION_KEY = "my_team"

# Team-perspective palette (win / draw / loss) + factor-badge accents. Mirrors the app's
# sky/slate/rose family so the page feels of a piece with the hero and bracket.
_WIN = "#34d399"  # emerald
_DRAW = "#94a3b8"  # slate
_LOSS = "#fb7185"  # rose

_BADGES = {
    "strength": ("🟢", "Strength", "#34d399"),
    "neutral": ("⚪", "Around field average", "#94a3b8"),
    "weakness": ("🔴", "Relative weakness", "#fb7185"),
    "na": ("⚠️", "Limited coverage", "#f59e0b"),
}
# Funnel colours, R32 → Champion (sky deepening to gold as the field narrows).
_FUNNEL = ["#38bdf8", "#22d3ee", "#34d399", "#a3e635", "#fbbf24", "#f59e0b"]


def _esc(text) -> str:
    return html.escape(str(text)) if text is not None else ""


# --------------------------------------------------------------------------------------
# Next-match prediction (the one model call this page makes)
# --------------------------------------------------------------------------------------
def predict_for_team(predictor, artifact, explainer, wc, fixture: dict, team: str) -> dict:
    """Calibrated win/draw/loss for ``team`` in ``fixture``, oriented to the team's perspective.

    Reuses the same host/venue convention as the rest of the app (:func:`components.home.match_context`)
    so the numbers match the Match Predictor and Matchday hero exactly, then flips them to read as the
    chosen team's win / draw / loss. Also surfaces the rest-days the model is using for each side and
    the top SHAP drivers, so the per-match read stays explainable.
    """
    t1, t2 = fixture["team1"], fixture["team2"]
    opp = t2 if team == t1 else t1
    home, away, neutral, is_host = home_view.match_context(wc, t1, t2, fixture.get("group"))
    feats = predictor.features(home, away, neutral=neutral, is_host_home=is_host)
    probs = predict_from_features(artifact, feats, neutral=neutral)
    ex = explain.explain_match(artifact, feats, probs, home, away, explainer)
    if home == team:
        win, draw, loss = probs["H"], probs["D"], probs["A"]
        rest_team, rest_opp = feats["days_rest_home"], feats["days_rest_away"]
    else:
        win, draw, loss = probs["A"], probs["D"], probs["H"]
        rest_team, rest_opp = feats["days_rest_away"], feats["days_rest_home"]
    drivers = [home_view._driver_label(d) for d in ex["contributions"] if d["shap"] > 1e-6][:3]
    return {
        "opponent": opp,
        "win": win,
        "draw": draw,
        "loss": loss,
        "neutral": neutral,
        "is_host": bool(is_host),
        "rest_team": rest_team,
        "rest_opp": rest_opp,
        "favored": ex["favored"],
        "win_prob": ex["win_prob"],
        "drivers": drivers,
    }


# --------------------------------------------------------------------------------------
# Hero
# --------------------------------------------------------------------------------------
def _hero(team, group, elo, elo_rank, n, champ_p, fav_p, host) -> None:
    host_badge = (
        f' · <span style="color:#fbbf24">🏠 Host — plays Group {host["group"]} at home</span>'
        if host
        else ""
    )
    if fav_p > 0 and champ_p >= fav_p - 1e-9:
        odds_line = (
            f"<b style='color:#fde68a'>{champ_p:.1%}</b> to lift the trophy — the model's outright "
            "favourite, though most simulated tournaments still end with someone else."
        )
    else:
        odds_line = (
            f"<b style='color:#fde68a'>{champ_p:.1%}</b> to lift the trophy "
            f"<span style='color:#94a3b8'>· field favourite tops out around {fav_p:.0%}</span> — "
            "the model spreads the title across all 48 teams."
        )
    body = (
        '<div style="border-radius:18px;padding:20px 24px;color:#f8fafc;'
        "background:linear-gradient(135deg,#0b1220 0%,#111c34 55%,#0b1220 100%);"
        "background-color:#0b1220;border:1px solid rgba(148,163,184,.18);"
        'box-shadow:0 12px 32px rgba(2,6,23,.45)">'
        '<div style="font-size:12px;font-weight:800;letter-spacing:.12em;text-transform:uppercase;'
        f'color:#7dd3fc">FIFA World Cup 2026 · Group {_esc(group)}</div>'
        f'<div style="font-size:38px;font-weight:800;line-height:1.1;margin-top:4px">'
        f"{_esc(team)}</div>"
        f'<div style="margin-top:6px;color:#94a3b8;font-size:14px;font-weight:600">Elo {elo:.0f} · '
        f"#{elo_rank} of {n} by rating{host_badge}</div>"
        f'<div style="margin-top:12px;font-size:15px;line-height:1.45">{odds_line}</div>'
        "</div>"
    )
    st.markdown(body, unsafe_allow_html=True)


# --------------------------------------------------------------------------------------
# Stage run funnel
# --------------------------------------------------------------------------------------
def _stage_funnel(stages: list[dict]) -> go.Figure:
    labels = [s["label"] for s in stages]
    vals = [s["p"] * 100 for s in stages]
    fig = go.Figure(
        go.Funnel(
            y=labels,
            x=vals,
            orientation="h",
            text=[f"{s['p']:.0%}" for s in stages],
            textposition="inside",
            textinfo="text",
            marker={"color": _FUNNEL[: len(stages)]},
            connector={"line": {"color": "rgba(148,163,184,.35)", "width": 1}},
            hovertemplate="%{y}: %{x:.1f}% of simulations<extra></extra>",
        )
    )
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
        yaxis=dict(title=None),
        xaxis=dict(visible=False),
    )
    return fig


# --------------------------------------------------------------------------------------
# Group outlook
# --------------------------------------------------------------------------------------
def _render_group(outlook: dict, team: str) -> None:
    st.markdown(f"#### 🪜 Group {outlook['group']} outlook")
    standings = outlook.get("standings")
    if standings:
        rows = [
            {
                "": "⭐" if s["team"] == team else "",
                "Team": s["team"],
                "Pl": s["played"],
                "Pts": s["pts"],
                "GD": s["gd"],
                "GF": s["gf"],
            }
            for s in standings
        ]
        st.caption("Current standings (played group matches):")
        st.dataframe(rows, hide_index=True, width="stretch")
    else:
        st.caption("No group games played yet — showing the model's projection.")

    st.caption("Model odds to advance to the knockout stage:")
    rivals = [
        {
            "": "⭐" if r["is_self"] else "",
            "Team": r["team"],
            "Elo": round(r["elo"]),
            "Advance": r["advance_p"],
        }
        for r in outlook["rivals"]
    ]
    st.dataframe(
        rivals,
        hide_index=True,
        width="stretch",
        column_config={
            "Advance": st.column_config.ProgressColumn(
                "Advance", format="percent", min_value=0.0, max_value=1.0
            )
        },
    )


# --------------------------------------------------------------------------------------
# Next match
# --------------------------------------------------------------------------------------
def _wdl_bar(team: str, pred: dict) -> str:
    win, draw, loss = pred["win"], pred["draw"], pred["loss"]
    return (
        '<div style="display:flex;justify-content:space-between;font-size:13px;font-weight:700;'
        'margin-bottom:5px">'
        f'<span style="color:{_WIN}">{_esc(team)} {win:.0%}</span>'
        f'<span style="color:#cbd5e1">Draw {draw:.0%}</span>'
        f'<span style="color:{_LOSS}">{_esc(pred["opponent"])} {loss:.0%}</span></div>'
        '<div style="display:flex;height:12px;border-radius:999px;overflow:hidden;'
        'background:rgba(148,163,184,.18)">'
        f'<div style="width:{win * 100:.1f}%;background:{_WIN}"></div>'
        f'<div style="width:{draw * 100:.1f}%;background:{_DRAW}"></div>'
        f'<div style="width:{loss * 100:.1f}%;background:{_LOSS}"></div></div>'
    )


def _render_next_match(predictor, artifact, explainer, wc, team, fixture, tz) -> None:
    st.markdown("#### ⚽ Next match")
    if fixture is None:
        st.caption(
            f"No upcoming group fixture for {team} in the schedule feed — the group stage may be "
            "done, or knockout opponents aren't decided yet. The run odds above already fold in "
            "every remaining match."
        )
        return
    pred = predict_for_team(predictor, artifact, explainer, wc, fixture, team)
    ko = None
    if fixture.get("kickoff_utc"):
        try:
            ko = datetime.fromisoformat(fixture["kickoff_utc"]).astimezone(tz)
        except (ValueError, TypeError):
            ko = None
    when = ko.strftime("%a %d %b · %H:%M %Z") if ko else (fixture.get("date") or "date TBD")
    venue = fixture.get("venue") or ""
    meta = " · ".join(filter(None, [when, venue]))
    st.markdown(f"**{_esc(team)} vs {_esc(pred['opponent'])}**")
    if meta:
        st.caption(meta)
    st.markdown(_wdl_bar(team, pred), unsafe_allow_html=True)
    venue_note = (
        "at home (host)" if pred["is_host"] else ("neutral venue" if pred["neutral"] else "")
    )
    rest = f"Rest: {team} {pred['rest_team']:.0f}d · {pred['opponent']} {pred['rest_opp']:.0f}d"
    st.caption(" · ".join(filter(None, [rest, venue_note])))
    if pred["drivers"]:
        st.caption("Model drivers: " + ", ".join(f"↑ {d}" for d in pred["drivers"]))


# --------------------------------------------------------------------------------------
# Strengths & weaknesses
# --------------------------------------------------------------------------------------
def _factor_card(f: dict) -> None:
    icon, badge_label, color = _BADGES[f["kind"]]
    with st.container(border=True):
        st.markdown(f"**{_esc(f['label'])}**")
        st.markdown(
            f"<div style='font-size:22px;font-weight:800;color:#e2e8f0'>{_esc(f['value_str'])}</div>",
            unsafe_allow_html=True,
        )
        if f["kind"] == "na":
            st.caption(f["note"])
            st.markdown(
                f"<span style='font-size:12px;font-weight:700;color:{color}'>{icon} {badge_label}"
                "</span>",
                unsafe_allow_html=True,
            )
            return
        pct = f["percentile"]
        st.markdown(
            "<div style='height:8px;border-radius:999px;background:rgba(148,163,184,.18);"
            "overflow:hidden;margin:4px 0 6px'>"
            f"<div style='width:{pct * 100:.0f}%;height:100%;background:{color}'></div></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<span style='font-size:12px;font-weight:700;color:{color}'>{icon} {badge_label}</span>"
            f"<span style='font-size:12px;color:#94a3b8'> · #{f['rank']} of {f['n']}</span>",
            unsafe_allow_html=True,
        )
        st.caption(f["note"])


def _render_factors(factors: list[dict], team: str) -> None:
    st.markdown("#### 🧬 Strengths & weaknesses, by the model")
    st.caption(
        f"Where {team} ranks among the 48 finalists on each input the model actually uses. "
        "Elo is the dominant signal; FIFA points and squad strength closely track it and add little "
        "measured lift (see **Under the Hood**) — so this is honest signal, not invented analysis."
    )
    per_row = 3
    for start in range(0, len(factors), per_row):
        chunk = factors[start : start + per_row]
        cols = st.columns(per_row)
        for col, f in zip(cols, chunk, strict=False):
            with col:
                _factor_card(f)


# --------------------------------------------------------------------------------------
# Bracket path
# --------------------------------------------------------------------------------------
def _render_path(stages: list[dict], team: str) -> None:
    st.markdown("#### 🗺️ Road through the bracket")
    chips = []
    arrow = '<span style="color:#475569;font-size:16px;align-self:center;margin:0 2px">→</span>'
    for i, s in enumerate(stages):
        intensity = 0.18 + 0.5 * s["p"]
        chips.append(
            '<div style="min-width:96px;text-align:center;padding:8px 10px;border-radius:11px;'
            f'background:rgba(56,189,248,{intensity:.2f});border:1px solid rgba(56,189,248,.35)">'
            f'<div style="font-size:11px;font-weight:700;color:#cbd5e1;text-transform:uppercase;'
            f'letter-spacing:.04em">{_esc(s["label"])}</div>'
            f'<div style="font-size:18px;font-weight:800;color:#f8fafc">{s["p"]:.0%}</div></div>'
        )
        if i < len(stages) - 1:
            chips.append(arrow)
    st.markdown(
        '<div style="display:flex;flex-wrap:wrap;gap:6px;align-items:stretch">'
        + "".join(chips)
        + "</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"Each step is the share of simulations in which {team} is still alive at that round — the "
        "drop-off is the model's uncertainty made visible."
    )


# --------------------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------------------
def render_team_page(
    predictor,
    artifact,
    explainer,
    wc,
    state,
    result,
    *,
    fixtures_list=None,
    played=None,
    now=None,
    tz=None,
    known_results=None,
    source=None,
    fetched_at=None,
) -> None:
    """Render the full Choose-Your-Team page for the team held in ``st.session_state``."""
    st.subheader("⭐ Choose Your Team")
    st.caption(
        "Pick a team and follow its tournament — run odds, group outlook, the next match, what the "
        "model rates and doubts, and what needs to happen. Probabilities are calibrated, not "
        "predictions."
    )

    if result is None:
        st.info("Run a tournament simulation first (the **Simulate** tab), then come back here.")
        return

    teams = sorted(tp.wc_teams(wc))
    default = "Brazil" if "Brazil" in teams else teams[0]
    st.session_state.setdefault(SESSION_KEY, default)
    # Plain country names (no flag): emoji flags render as bare letters on Windows ("BR Brazil")
    # and as a generic black flag for England/Scotland, and a native selectbox can't show real
    # flag images — so we match the Match Predictor and show the name alone.
    st.selectbox(
        "Your team",
        teams,
        key=SESSION_KEY,
    )
    team = st.session_state[SESSION_KEY]

    now = now or datetime.now(UTC)
    dossier = tp.build_dossier(
        wc, result, state, fixtures_list or [], team, now, known_results=known_results
    )
    if not dossier["in_field"]:
        st.warning(f"{team} is not in the 2026 field.")
        return

    stages = dossier["stages"]
    champ_p = next((s["p"] for s in stages if s["key"] == "Champion"), 0.0)
    field = tp.wc_teams(wc)
    elo = state.ratings.get(team, 1500.0)
    elo_rank, n, _ = tp._rank_percentile(elo, [state.ratings.get(t, 1500.0) for t in field])

    _hero(
        team,
        dossier["group"],
        elo,
        elo_rank,
        n,
        champ_p,
        dossier["champion_field_max"],
        dossier["host"],
    )

    n_sims = dossier.get("n_sims")
    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### 📈 How far the model sees them going")
        st.plotly_chart(_stage_funnel(stages), width="stretch")
        if n_sims:
            st.caption(
                f"Calibrated probabilities across {n_sims:,} simulated tournaments — the numbers "
                "are the uncertainty, not a forecast of one outcome."
            )
    with right:
        _render_next_match(predictor, artifact, explainer, wc, team, dossier["next_fixture"], tz)
        st.markdown("")
        _render_group(dossier["outlook"], team)

    st.divider()
    _render_path(stages, team)
    with st.expander(f"See {team}'s road traced on the full bracket", expanded=False):
        bracket_view.render_legend(team)
        bracket_view.render_bracket(result, state.ratings, selected_team=team, played=played)

    st.divider()
    _render_factors(dossier["factors"], team)

    st.divider()
    st.markdown("#### 🧭 What needs to happen?")
    for line in dossier["what_needs_to_happen"]:
        st.markdown(f"- {line}")

    if source:
        as_of = f" · as of {fetched_at}" if fetched_at else ""
        st.caption(f"Simulation standings source: {source}{as_of}")
