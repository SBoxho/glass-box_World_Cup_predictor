"""Matchday Home — the landing page.

A visually-led "what's happening at the World Cup right now" screen that fronts the data-science
tabs. It answers, in order of relevance to *this moment*:

* a **hero** for the most relevant match — live now, or a ticking countdown to the next one (with
  the model's prediction + top SHAP drivers once it is inside 24 hours);
* the **today's matches** and **recently finished** strips (rendered by :mod:`components.matchday`);
* three **action cards** routing to the Match Predictor, Tournament Simulator, and bracket.

This module is pure presentation: it resolves one :class:`core.fixtures.MatchdayContext` for the
moment, asks the existing :class:`core.model.Predictor` / :mod:`core.explain` for predictions, and
renders (handing the context's strips to :mod:`components.matchday`). The ticking clocks are
client-side JS (so the page feels live without server reruns); everything degrades gracefully when
the feed is offline, empty, or the tournament is over — the schedule itself is committed, so the
page always has something to show. The tone stays honest throughout — calibrated probabilities, not
betting tips.
"""

from __future__ import annotations

import html
from datetime import UTC, datetime

import streamlit as st
import streamlit.components.v1 as components

from components import matchday
from core import explain
from core import fixtures as fx
from core.model import predict_from_features

# Navigation view labels — shared with streamlit_app's segmented-control nav so the action-card
# callbacks can switch tabs by setting the same session-state key.
VIEW_MATCHDAY = "🏟️ Matchday"
VIEW_PREDICT = "🎯 Predict a Match"
VIEW_SIMULATE = "🏆 Simulate"
VIEW_TEAM = "⭐ My Team"
VIEW_BRACKET = "🔬 Under the Hood"
NAV_KEY = "nav"

# Hero colour scheme (self-contained inside the iframe — does not inherit the Streamlit theme).
_HOME_COLOR = "#38bdf8"  # team1 / home — sky
_DRAW_COLOR = "#94a3b8"  # draw — slate
_AWAY_COLOR = "#fb7185"  # team2 / away — rose


# --------------------------------------------------------------------------------------
# Timezone + match-context helpers
# --------------------------------------------------------------------------------------
def resolve_timezone() -> tuple[object, str]:
    """Best-effort user timezone from the browser, falling back to Europe/Luxembourg.

    Streamlit ≥1.40 exposes the browser's IANA zone via ``st.context.timezone``. We read it
    defensively (older runtimes / headless contexts have no ``context.timezone``) and hand the name
    to :func:`core.fixtures.resolve_timezone`, which owns the fallback chain (the browser zone →
    Europe/Luxembourg → UTC) so the home page renders sensible local kickoff times wherever it runs.
    """
    try:
        name = st.context.timezone
    except AttributeError:  # never let tz lookup block render
        name = None
    return fx.resolve_timezone(name)


def match_context(
    wc: dict, team1: str, team2: str, group: str | None
) -> tuple[str, str, bool, int]:
    """Return ``(home, away, neutral, is_host_home)`` using the simulator's host convention.

    Mirrors :meth:`core.simulate.TournamentSimulator._group_fixtures`: in a group containing a
    2026 host, that host is the home side of its matches (non-neutral); every other group match —
    and every knockout tie — is played at a neutral venue. Keeps home-page predictions identical to
    what the Tournament Simulator would compute for the same fixture.
    """
    host_groups = wc.get("host_groups", {})
    if group:
        for host, g in host_groups.items():
            if g == group and host in (team1, team2):
                if host == team1:
                    return team1, team2, False, 1
                return team2, team1, False, 1
    return team1, team2, True, 0


def _wc_team_set(wc: dict) -> set[str]:
    return {t for teams in wc["groups"].values() for t in teams}


def _driver_label(contribution: dict) -> str:
    """A display label for one SHAP driver, prettifying any raw feature name as a fallback.

    ``core.explain.FEATURE_LABELS`` humanizes every model feature; this only kicks in if a future
    feature ships without a label (the contribution's ``label`` would then be the raw name).
    """
    feature, label = contribution.get("feature"), contribution.get("label", "")
    return label.replace("_", " ") if label == feature else label


def predict_fixture(predictor, artifact, explainer, wc, fixture: dict, wc_teams: set[str]):
    """Calibrated prediction + top drivers for a fixture, oriented to ``team1``/``team2``.

    Returns ``None`` when either side is not one of the 48 drawn teams (e.g. a knockout slot whose
    teams are not decided yet) — the hero then shows the countdown without a spurious prediction.
    """
    t1, t2 = fixture["team1"], fixture["team2"]
    if t1 not in wc_teams or t2 not in wc_teams:
        return None
    home, away, neutral, is_host = match_context(wc, t1, t2, fixture["group"])
    feats = predictor.features(home, away, neutral=neutral, is_host_home=is_host)
    probs = predict_from_features(artifact, feats, neutral=neutral)  # {H, D, A} (home/away)
    ex = explain.explain_match(artifact, feats, probs, home, away, explainer)
    if home == t1:
        p1, pdw, p2 = probs["H"], probs["D"], probs["A"]
    else:
        p1, pdw, p2 = probs["A"], probs["D"], probs["H"]
    drivers = [_driver_label(d) for d in ex["contributions"] if d["shap"] > 1e-6][:3]
    return {
        "team1": t1,
        "team2": t2,
        "p1": p1,
        "pd": pdw,
        "p2": p2,
        "favored": ex["favored"],
        "win_prob": ex["win_prob"],
        "drivers": drivers,
    }


# --------------------------------------------------------------------------------------
# Small formatting helpers
# --------------------------------------------------------------------------------------
def _fmt_local(dt_utc: datetime, tz) -> str:
    """Format a UTC kickoff in the user's zone, e.g. ``"Thu 18 Jun · 21:00 CEST"``."""
    local = dt_utc.astimezone(tz)
    abbr = local.strftime("%Z")
    return f"{local.strftime('%a %d %b · %H:%M')} {abbr}".strip()


def _meta_line(fixture: dict) -> str:
    parts: list[str] = []
    if fixture.get("group"):
        parts.append(f"Group {fixture['group']}")
    elif fixture.get("round"):
        parts.append(str(fixture["round"]))
    if fixture.get("venue"):
        parts.append(str(fixture["venue"]))
    return " · ".join(parts)


# --------------------------------------------------------------------------------------
# Hero card (HTML/CSS in an iframe, with client-side ticking clocks)
# --------------------------------------------------------------------------------------
_CARD_CSS = """
<style>
  html, body { margin: 0; background: transparent; }
  * { box-sizing: border-box; }
  .hero {
    font-family: "Source Sans Pro", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    color: #f8fafc;
    background:
      radial-gradient(1200px 200px at 15% -40%, rgba(56,189,248,.22), transparent 60%),
      linear-gradient(135deg, #0b1220 0%, #111c34 55%, #0b1220 100%);
    border: 1px solid rgba(148,163,184,.18);
    border-radius: 18px;
    padding: 20px 24px;
    box-shadow: 0 12px 32px rgba(2,6,23,.45);
  }
  .badge {
    display: inline-flex; align-items: center; gap: 8px;
    font-size: 12px; font-weight: 800; letter-spacing: .12em; text-transform: uppercase;
    padding: 5px 12px; border-radius: 999px;
    background: rgba(56,189,248,.14); color: #7dd3fc; border: 1px solid rgba(56,189,248,.35);
  }
  .badge.live { background: rgba(244,63,94,.16); color: #fda4af; border-color: rgba(244,63,94,.45); }
  .dot { width: 9px; height: 9px; border-radius: 50%; background: #f43f5e;
         box-shadow: 0 0 0 0 rgba(244,63,94,.7); animation: pulse 1.4s infinite; }
  @keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(244,63,94,.6); }
    70% { box-shadow: 0 0 0 10px rgba(244,63,94,0); }
    100% { box-shadow: 0 0 0 0 rgba(244,63,94,0); }
  }
  .meta { margin-top: 12px; color: #94a3b8; font-size: 13.5px; font-weight: 600;
          letter-spacing: .02em; }
  .teams { margin-top: 6px; display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }
  .team { font-size: 30px; font-weight: 800; line-height: 1.05; }
  .vs { color: #64748b; font-size: 16px; font-weight: 700; }
  .clock { margin-top: 14px; font-variant-numeric: tabular-nums;
           font-size: 40px; font-weight: 800; letter-spacing: .01em; color: #e2e8f0; }
  .clock.cd { color: #7dd3fc; }
  .clock.live { color: #fda4af; }
  .score { margin-top: 12px; font-size: 40px; font-weight: 800; }
  .sub { color: #94a3b8; font-size: 13px; margin-top: 2px; }
  .probs { margin-top: 16px; }
  .plabels { display: flex; justify-content: space-between; font-size: 13px; font-weight: 700;
             margin-bottom: 6px; }
  .plabels .draw { color: #cbd5e1; }
  .bar { display: flex; height: 12px; border-radius: 999px; overflow: hidden;
         background: rgba(148,163,184,.18); }
  .seg { height: 100%; }
  .drivers { margin-top: 14px; display: flex; flex-wrap: wrap; gap: 8px; }
  .chip { font-size: 12px; font-weight: 700; padding: 4px 10px; border-radius: 999px;
          background: rgba(34,197,94,.14); color: #86efac; border: 1px solid rgba(34,197,94,.3); }
  .scorers { margin-top: 10px; color: #cbd5e1; font-size: 12.5px; }
  .foot { margin-top: 16px; color: #8aa0bd; font-size: 12px; font-style: italic; }
</style>
"""

_CARD_JS = """
<script>
(function () {
  function pad(n) { return String(n).padStart(2, "0"); }
  // Auto-fit the component iframe to the card's real height. A single fixed height can't suit
  // both desktop (teams on one line) and mobile (everything wraps); this keeps it tight and
  // un-clipped at any width. Same-origin srcdoc iframe, so frameElement is reachable.
  function fit() {
    try {
      var h = document.body.scrollHeight;
      if (window.frameElement) { window.frameElement.style.height = (h + 2) + "px"; }
    } catch (e) { /* cross-origin or no frame — fall back to the server-set height */ }
  }
  fit();
  window.addEventListener("load", fit);
  window.addEventListener("resize", fit);
  setTimeout(fit, 60); setTimeout(fit, 300);
  var cd = document.getElementById("cd");
  if (cd) {
    var target = new Date(cd.dataset.target).getTime();
    var tickCd = function () {
      var diff = target - Date.now();
      if (diff <= 0) { cd.textContent = "Kicking off…"; return; }
      var s = Math.floor(diff / 1000);
      var d = Math.floor(s / 86400); s -= d * 86400;
      var h = Math.floor(s / 3600); s -= h * 3600;
      var m = Math.floor(s / 60); s -= m * 60;
      cd.textContent = (d > 0 ? d + "d " : "") + pad(h) + ":" + pad(m) + ":" + pad(s);
    };
    tickCd(); setInterval(tickCd, 1000);
  }
  var mins = document.getElementById("mins");
  if (mins) {
    var kick = new Date(mins.dataset.kick).getTime();
    var tickMin = function () {
      var m = Math.max(0, Math.floor((Date.now() - kick) / 60000));
      mins.textContent = "~" + m + "′";
    };
    tickMin(); setInterval(tickMin, 1000);
  }
})();
</script>
"""


def _esc(text) -> str:
    return html.escape(str(text)) if text is not None else ""


def _prob_bar_html(pred: dict) -> str:
    return (
        '<div class="probs">'
        '<div class="plabels">'
        f'<span style="color:{_HOME_COLOR}">{_esc(pred["team1"])} {pred["p1"]:.0%}</span>'
        f'<span class="draw">Draw {pred["pd"]:.0%}</span>'
        f'<span style="color:{_AWAY_COLOR}">{_esc(pred["team2"])} {pred["p2"]:.0%}</span>'
        "</div>"
        '<div class="bar">'
        f'<div class="seg" style="width:{pred["p1"] * 100:.1f}%;background:{_HOME_COLOR}"></div>'
        f'<div class="seg" style="width:{pred["pd"] * 100:.1f}%;background:{_DRAW_COLOR}"></div>'
        f'<div class="seg" style="width:{pred["p2"] * 100:.1f}%;background:{_AWAY_COLOR}"></div>'
        "</div></div>"
    )


def _drivers_html(pred: dict) -> str:
    chips = "".join(f'<span class="chip">↑ {_esc(d)}</span>' for d in pred.get("drivers", []))
    return f'<div class="drivers">{chips}</div>' if chips else ""


def _teams_html(team1: str, team2: str) -> str:
    return (
        '<div class="teams">'
        f'<span class="team">{_esc(team1)}</span>'
        '<span class="vs">vs</span>'
        f'<span class="team">{_esc(team2)}</span>'
        "</div>"
    )


def _render_card(body: str, height: int) -> None:
    components.html(_CARD_CSS + f'<div class="hero">{body}</div>' + _CARD_JS, height=height)


def _scorers_html(fixture: dict) -> str:
    s1, s2 = fixture.get("scorers1") or [], fixture.get("scorers2") or []
    if not s1 and not s2:
        return ""
    line = " · ".join(filter(None, ["; ".join(s1), "; ".join(s2)]))
    return f'<div class="scorers">⚽ {_esc(line)}</div>'


def _hero_live(fixture: dict, pred: dict | None, tz) -> None:
    """Hero for a match in progress (inferred live — no minute-by-minute feed)."""
    ko = fx.kickoff_datetime(fixture)
    has_score = fixture.get("team1_score") is not None
    if has_score:  # a "recent" match the feed already scored, surfaced as the live headline
        center = (
            f'<div class="score">{fixture["team1_score"]} – {fixture["team2_score"]}</div>'
            '<div class="sub">latest from the live feed</div>'
        )
    else:
        center = (
            f'<div class="clock live" id="mins" data-kick="{_esc(ko.isoformat())}">~0′</div>'
            '<div class="sub">approx. elapsed · live score posts when the feed updates</div>'
        )
    body = (
        '<div class="badge live"><span class="dot"></span>Live now</div>'
        f'<div class="meta">{_esc(_meta_line(fixture))}</div>'
        + _teams_html(fixture["team1"], fixture["team2"])
        + center
        + _scorers_html(fixture)
    )
    height = 270
    if pred:
        body += _prob_bar_html(pred) + _drivers_html(pred)
        body += (
            f'<div class="foot">Pre-match model leans {_esc(pred["favored"])} '
            f"({pred['win_prob']:.0%} to win) — calibrated probabilities, not betting advice.</div>"
        )
        height = 400 if pred.get("drivers") else 360
    _render_card(body, height)


def _hero_next(fixture: dict, pred: dict | None, tz, within_24h: bool) -> None:
    """Hero for the next upcoming match: a ticking countdown, plus the prediction within 24h."""
    ko = fx.kickoff_datetime(fixture)
    label = "Next match" if not within_24h else "Up next · within 24h"
    body = (
        f'<div class="badge">⏱ {label}</div>'
        f'<div class="meta">{_esc(_meta_line(fixture))}</div>'
        + _teams_html(fixture["team1"], fixture["team2"])
        + f'<div class="clock cd" id="cd" data-target="{_esc(ko.isoformat())}">--:--:--</div>'
        + f'<div class="sub">until kick-off · {_esc(_fmt_local(ko, tz))}</div>'
    )
    height = 250
    if within_24h and pred:
        body += _prob_bar_html(pred) + _drivers_html(pred)
        body += (
            f'<div class="foot">Model leans {_esc(pred["favored"])} '
            f"({pred['win_prob']:.0%} to win) — calibrated probabilities, not betting advice.</div>"
        )
        height = 400 if pred.get("drivers") else 360
    else:
        body += (
            '<div class="foot">Prediction & explanation appear here once kick-off is within '
            "24 hours.</div>"
        )
    _render_card(body, height)


def _hero_message(badge: str, title: str, sub: str, height: int = 200) -> None:
    """Generic fallback hero (no schedule, tournament finished, etc.)."""
    body = (
        f'<div class="badge">{_esc(badge)}</div>'
        f'<div class="teams"><span class="team">{_esc(title)}</span></div>'
        f'<div class="sub" style="margin-top:10px;font-size:14px">{_esc(sub)}</div>'
    )
    _render_card(body, height)


def render_hero(predictor, artifact, explainer, wc, snapshot, now, tz) -> None:
    """Pick and render the single most relevant hero for the current moment."""
    fixtures = (snapshot or {}).get("fixtures") or []
    wc_teams = _wc_team_set(wc)

    # Fallback 1 — no schedule at all (offline first run, source unavailable).
    if not fixtures:
        start, end = wc.get("dates", {}).get("start"), wc.get("dates", {}).get("end")
        window = f"{start} → {end}" if start and end else "summer 2026"
        _hero_message(
            "Schedule unavailable",
            wc.get("tournament", "FIFA World Cup 2026"),
            f"Couldn't reach the live schedule feed. Tournament window: {window}. "
            "Use the tools below — they work fully offline.",
            height=210,
        )
        return

    # 1) A match in progress wins the hero.
    live = fx.current_match(fixtures, now)
    if live is not None:
        pred = predict_fixture(predictor, artifact, explainer, wc, live, wc_teams)
        _hero_live(live, pred, tz)
        return

    # 2) Otherwise count down to the next match.
    nxt = fx.next_match(fixtures, now)
    if nxt is not None:
        ko = fx.kickoff_datetime(nxt)
        within_24h = (ko - now).total_seconds() <= 24 * 3600
        pred = (
            predict_fixture(predictor, artifact, explainer, wc, nxt, wc_teams)
            if within_24h
            else None
        )
        _hero_next(nxt, pred, tz, within_24h)
        return

    # 3) No live and nothing upcoming → the tournament is over (or the feed is exhausted).
    finished = [f for f in fixtures if f.get("finished")]
    last = max(finished, key=lambda f: fx.kickoff_datetime(f) or now) if finished else None
    if last is not None:
        score = f"{last['team1']} {last['team1_score']}–{last['team2_score']} {last['team2']}"
        _hero_message(
            "Tournament complete",
            "That's a wrap on World Cup 2026",
            f"Last result on file: {score}. Explore the run with the tools below.",
            height=210,
        )
    else:
        _hero_message(
            "No matches scheduled",
            wc.get("tournament", "FIFA World Cup 2026"),
            "No live or upcoming matches in the feed right now.",
            height=200,
        )


# --------------------------------------------------------------------------------------
# Action cards
# --------------------------------------------------------------------------------------
def _goto(view: str) -> None:
    """on_click callback: switch the top-level nav (safe to set a widget key from a callback)."""
    st.session_state[NAV_KEY] = view


def render_action_cards() -> None:
    st.markdown("#### Dig into the model")
    cards = [
        (
            "🎯",
            "Predict a Match",
            "Any two teams, any venue — calibrated odds with a SHAP read.",
            VIEW_PREDICT,
        ),
        (
            "🏆",
            "Simulate the Tournament",
            "Monte-Carlo the whole 48-team bracket to title odds.",
            VIEW_SIMULATE,
        ),
        (
            "⭐",
            "Choose Your Team",
            "Follow one team — run odds, group, next match, what needs to happen.",
            VIEW_TEAM,
        ),
        (
            "🧭",
            "Under the Hood",
            "How the model is built, calibrated, and where it falls short.",
            VIEW_BRACKET,
        ),
    ]
    cols = st.columns(len(cards))
    for col, (icon, title, blurb, view) in zip(cols, cards, strict=True):
        with col.container(border=True):
            st.markdown(f"### {icon} {title}")
            st.caption(blurb)
            st.button(
                f"{title} →",
                key=f"action_{view}",
                width="stretch",
                on_click=_goto,
                args=(view,),
            )


# --------------------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------------------
def render_home(predictor, artifact, explainer, wc, state, snapshot) -> None:
    """Render the full Matchday Home page from a schedule ``snapshot`` (may be ``None``/empty)."""
    tz, tz_label = resolve_timezone()
    now = datetime.now(UTC)

    head_l, head_r = st.columns([3, 1])
    with head_l:
        st.markdown("### ⚽ Matchday")
        st.caption(
            "Your live read on World Cup 2026 — calibrated model predictions, explained. "
            "Probabilistic, not betting advice."
        )
    with head_r:
        if st.button("↻ Refresh", width="stretch", help="Re-pull the live schedule feed"):
            st.session_state["fixtures_nonce"] = st.session_state.get("fixtures_nonce", 0) + 1
            st.rerun()

    # One resolve of the full matchday picture drives both the hero and the strips below.
    fixtures_list = (snapshot or {}).get("fixtures") or []
    ctx = fx.get_matchday_context(now, tz, fixtures=fixtures_list)

    render_hero(predictor, artifact, explainer, wc, snapshot, now, tz)

    # Freshness / source line — honest about offline/cached state.
    if snapshot and snapshot.get("error") and not (snapshot.get("fixtures")):
        st.caption("⚠️ Live schedule unavailable (offline or source down) — showing fallbacks.")
    elif snapshot and snapshot.get("fetched_at"):
        st.caption(
            f"📡 Schedule as of {snapshot['fetched_at']} · source: {snapshot.get('source')} · "
            f"times shown in {tz_label}"
        )

    st.divider()
    matchday.render_today(ctx)
    if ctx.recently_finished:
        st.divider()
        matchday.render_recently_finished(ctx)
    st.divider()
    render_action_cards()
