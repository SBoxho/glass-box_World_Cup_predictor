"""Matchday status panel — a small, self-contained renderer for "what's happening now".

This module is *pure presentation*. Every function takes the render-ready
:class:`core.fixtures.MatchdayContext` (or a slice of it) and draws it with plain Streamlit — no
network, no model calls, no schedule logic. All of that lives in :mod:`core.fixtures`; the panel
only reads the objects it is handed (plus the context's own ``now``/``timezone`` for status badges
and local kick-off formatting). It renders four sections, each skipped when empty:

* **🔴 Live now** — matches in progress (inferred — openfootball has no minute-by-minute feed).
* **⏱ Next match in Xh Ym** — the soonest kick-off, with the local date/time.
* **Today's matches** — every fixture on today's local date, with a status badge.
* **✅ Recently finished** — full-time results from the last couple of days.

:func:`render_matchday_panel` stacks all four. The richer hero card (ticking countdown + the
model's prediction) lives in :mod:`components.home`; this panel is the plain, reusable companion.
"""

from __future__ import annotations

from datetime import datetime, tzinfo

import streamlit as st

from core import fixtures as fx


# --------------------------------------------------------------------------------------
# Small formatting helpers (presentation only)
# --------------------------------------------------------------------------------------
def _meta_line(fixture: dict) -> str:
    """``"Group A · Mexico City"`` / ``"Round of 32 · Atlanta"`` — group-or-round, then venue."""
    parts: list[str] = []
    if fixture.get("group"):
        parts.append(f"Group {fixture['group']}")
    elif fixture.get("round"):
        parts.append(str(fixture["round"]))
    if fixture.get("venue"):
        parts.append(str(fixture["venue"]))
    return " · ".join(parts)


def _local_time(fixture: dict, tz: tzinfo) -> str:
    """Kick-off as a bare local ``HH:MM`` (``"TBD"`` if the fixture has no resolved time)."""
    ko = fx.kickoff_datetime(fixture)
    return ko.astimezone(tz).strftime("%H:%M") if ko else "TBD"


def _local_datetime(fixture: dict, tz: tzinfo) -> str:
    """Kick-off as ``"Thu 18 Jun · 21:00 CEST"`` in the user's zone (date-only fallback if no time)."""
    ko = fx.kickoff_datetime(fixture)
    if ko is None:
        return str(fixture.get("date") or "date TBD")
    local = ko.astimezone(tz)
    return f"{local.strftime('%a %d %b · %H:%M')} {local.strftime('%Z')}".strip()


def _score(fixture: dict) -> str:
    return f"{fixture['team1_score']}–{fixture['team2_score']}"


def _status_badge(fixture: dict, now: datetime, tz: tzinfo) -> str:
    """A compact status chip for the "today" grid, keyed off the coarse :func:`match_status`."""
    status = fx.match_status(fixture, now)
    if status == "finished":
        return f"🟢 FT {_score(fixture)}"
    if status == "live":
        return f"🔴 LIVE {_score(fixture)}" if fixture.get("team1_score") is not None else "🔴 LIVE"
    if status == "upcoming":
        return f"⏰ {_local_time(fixture, tz)}"
    return "🗓️ TBD"


def _matchup(fixture: dict) -> str:
    return f"{fixture['team1']}  \nvs  \n**{fixture['team2']}**"


# --------------------------------------------------------------------------------------
# The four sections (each a no-op when its slice of the context is empty)
# --------------------------------------------------------------------------------------
def render_live_now(ctx: fx.MatchdayContext) -> None:
    """🔴 Live now — one bordered card per in-progress match."""
    if not ctx.live:
        return
    st.markdown("#### 🔴 Live now")
    cols = st.columns(min(len(ctx.live), 3))
    for col, fixture in zip(cols, ctx.live, strict=False):
        with col.container(border=True):
            if fixture.get("team1_score") is not None:
                st.markdown(f"**{fixture['team1']} {_score(fixture)} {fixture['team2']}**")
            else:
                st.markdown(f"**{fixture['team1']} vs {fixture['team2']}**")
                st.caption("in progress · score posts when the feed updates")
            meta = _meta_line(fixture)
            if meta:
                st.caption(meta)


def render_next_match(ctx: fx.MatchdayContext) -> None:
    """⏱ Next match in Xh Ym — the soonest kick-off and its local date/time."""
    if ctx.next_match is None:
        return
    fixture = ctx.next_match
    st.markdown(f"#### ⏱ Next match in {ctx.next_in}")
    with st.container(border=True):
        st.markdown(f"**{fixture['team1']} vs {fixture['team2']}**")
        st.caption(f"Kick-off {_local_datetime(fixture, ctx.timezone)} ({ctx.tz_label})")
        meta = _meta_line(fixture)
        if meta:
            st.caption(meta)


def render_today(ctx: fx.MatchdayContext) -> None:
    """Today's matches — a grid of every fixture on today's local date, with a status badge."""
    st.markdown("#### Today's matches")
    if not ctx.today:
        st.caption("No matches on today's date. The next kick-off is shown above. ⤴")
        return
    per_row = 4
    for start in range(0, len(ctx.today), per_row):
        chunk = ctx.today[start : start + per_row]
        cols = st.columns(per_row)
        for col, fixture in zip(cols, chunk, strict=False):  # last row may be short
            with col.container(border=True):
                st.markdown(f"**{_status_badge(fixture, ctx.now, ctx.timezone)}**")
                st.markdown(_matchup(fixture))
                label = (
                    f"Group {fixture['group']}"
                    if fixture.get("group")
                    else (fixture.get("round") or "")
                )
                if label:
                    st.caption(label)


def render_recently_finished(ctx: fx.MatchdayContext) -> None:
    """✅ Recently finished — full-time results from the last couple of days, newest first."""
    if not ctx.recently_finished:
        return
    st.markdown("#### ✅ Recently finished")
    per_row = 4
    for start in range(0, len(ctx.recently_finished), per_row):
        chunk = ctx.recently_finished[start : start + per_row]
        cols = st.columns(per_row)
        for col, fixture in zip(cols, chunk, strict=False):
            with col.container(border=True):
                st.markdown(f"**{fixture['team1']} {_score(fixture)} {fixture['team2']}**")
                meta = _meta_line(fixture)
                if meta:
                    st.caption(meta)


def render_matchday_panel(ctx: fx.MatchdayContext) -> None:
    """Render all four sections in priority order (each skips itself when empty).

    A standalone, hero-free way to render the whole matchday picture from a single context — used
    on its own where the full prediction hero isn't wanted.
    """
    render_live_now(ctx)
    render_next_match(ctx)
    render_today(ctx)
    render_recently_finished(ctx)
