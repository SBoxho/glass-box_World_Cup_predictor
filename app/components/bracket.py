"""Knockout bracket view — a visually-led render of the simulator's per-slot projection.

Turns one :class:`core.simulate.SimulationResult` (specifically its ``bracket`` projection, built by
:mod:`core.bracket`) into an interactive Road-to-the-Final bracket: Round of 32 → Champion, each
knockout match a card showing its likely participants (with "X% to reach this slot" labels), the
flags, and the simulated head-to-head winner split. The four editorial highlights the brief asks for
— the projected champion, the biggest upset (Cinderella) path, the most uncertain match, and a
user-selected team's path — are traced as coloured roads through the tree.

Design choices, consistent with :mod:`components.home`:

* **Self-contained HTML/CSS in an iframe** (``components.html``) — no heavyweight JS framework. The
  styling lives inside the iframe (it does not inherit the Streamlit theme), matching the hero card.
* **Exact bracket geometry.** Each round's cards sit in fixed-height *bands* that double every round
  (R32 band = B, R16 = 2B, …), so a parent card's centre is exactly the midpoint of its two
  children — the classic funnel, pixel-aligned. A small post-layout JS pass draws the SVG connectors
  and auto-fits the iframe height; if it fails the funnel still reads on its own.
* **Responsive.** The bracket scrolls horizontally on narrow screens (laptops fit it; phones swipe
  through the rounds) and the iframe grows to its content height so the page — not an inner box —
  owns the vertical scroll.

This module is pure presentation: all aggregation/derivation is in ``core`` (and unit-tested there).
"""

from __future__ import annotations

import html

import streamlit as st
import streamlit.components.v1 as components

from core import bracket as cb
from core import flags

# Palette — self-contained inside the iframe (does not inherit the Streamlit theme). Mirrors the
# home hero's sky/slate/rose so the two screens feel like one app.
_TOP = "#38bdf8"  # top slot — sky
_BOT = "#fb7185"  # bottom slot — rose
_MUTE = "#94a3b8"

# Highlight accents (the four call-outs) + their priority for the card's primary (left) border.
_ACCENT = {
    "champ": "#fbbf24",  # projected champion — gold
    "upset": "#a78bfa",  # Cinderella run — violet
    "toss": "#f59e0b",  # most uncertain match — amber
    "sel": "#22d3ee",  # user-selected team — cyan
}
_PRIORITY = ["sel", "toss", "champ", "upset"]

# Per-round band height multiplier (2**round_index) — the geometry that aligns the funnel exactly.
_BAND_MULT = {"R32": 1, "R16": 2, "QF": 4, "SF": 8, "Final": 16}
_BAND_PX = 84  # height of a single Round-of-32 band; column height = 16 * this for every round


_CSS = """
<style>
  html, body { margin: 0; background: transparent; }
  * { box-sizing: border-box; }
  .wrap {
    font-family: "Source Sans Pro", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    color: #e2e8f0;
  }
  .scroll { overflow-x: auto; overflow-y: hidden; padding: 4px 2px 10px; }
  .bracket { position: relative; display: flex; align-items: stretch; width: max-content;
             min-width: 100%; }
  svg.links { position: absolute; top: 0; left: 0; pointer-events: none; z-index: 0; }
  .col { display: flex; flex-direction: column; min-width: 178px; width: 178px; margin-right: 34px;
         z-index: 1; }
  .col:last-child { margin-right: 0; }
  .col-head { text-align: center; font-weight: 800; font-size: 12.5px; letter-spacing: .06em;
              text-transform: uppercase; color: #cbd5e1; padding: 4px 0 2px; }
  .col-head .n { display: block; font-size: 10.5px; font-weight: 700; color: #64748b;
                 letter-spacing: .04em; }
  .col-body { display: flex; flex-direction: column; }
  .band { display: flex; align-items: center; justify-content: center; }
  .match {
    width: 100%; background: linear-gradient(180deg, #111c34 0%, #0e1830 100%);
    border: 1px solid rgba(148,163,184,.20); border-left: 3px solid rgba(148,163,184,.30);
    border-radius: 11px; padding: 7px 9px; box-shadow: 0 6px 16px rgba(2,6,23,.35);
  }
  .slot { display: flex; align-items: center; gap: 7px; height: 21px; }
  .slot .flag { font-size: 15px; width: 19px; text-align: center; flex: none; line-height: 1; }
  .slot .code { font-size: 9.5px; font-weight: 800; width: 19px; text-align: center; flex: none;
                color: #cbd5e1; background: rgba(148,163,184,.18); border-radius: 4px;
                padding: 1px 0; }
  .slot .name { font-size: 13px; font-weight: 700; white-space: nowrap; overflow: hidden;
                text-overflow: ellipsis; flex: 1 1 auto; min-width: 0; }
  .slot .pct { font-size: 11.5px; font-weight: 800; color: #cbd5e1; flex: none;
               font-variant-numeric: tabular-nums; }
  .slot.dim .name { color: #8aa0bd; font-weight: 600; }
  .slot.fav .name { color: #f8fafc; }
  .slot .sec { font-size: 9.5px; font-weight: 700; color: #64748b; margin-left: 4px;
               white-space: nowrap; }
  .slotlabel { font-size: 9px; font-weight: 800; letter-spacing: .04em; text-transform: uppercase;
               color: #475569; margin-left: 26px; line-height: 1.1; }
  .mid { display: flex; align-items: center; gap: 6px; margin: 3px 0; }
  .splitbar { flex: 1 1 auto; display: flex; height: 7px; border-radius: 999px; overflow: hidden;
              background: rgba(148,163,184,.16); }
  .seg { height: 100%; }
  .seg.rest { background: transparent; }
  .winpct { display: flex; justify-content: space-between; font-size: 9.5px; font-weight: 800;
            font-variant-numeric: tabular-nums; margin-top: 1px; }
  .winpct .l { color: #7dd3fc; } .winpct .r { color: #fda4af; }
  .played { font-size: 13px; font-weight: 800; text-align: center; color: #fde68a; margin: 2px 0; }
  .tags { display: flex; gap: 3px; justify-content: flex-end; height: 0; }
  .dot { width: 7px; height: 7px; border-radius: 50%; margin-top: -3px; }
  .tag { position: relative; font-size: 8.5px; font-weight: 800; letter-spacing: .03em;
         text-transform: uppercase; padding: 1px 5px; border-radius: 999px; margin-top: -4px; }
  .ftbadge { font-size: 8.5px; font-weight: 800; color: #86efac; letter-spacing: .04em; }
  .champ-card { background: linear-gradient(180deg, #2a2410 0%, #1a1606 100%);
                border: 1px solid rgba(251,191,36,.5); border-left: 3px solid #fbbf24;
                border-radius: 13px; padding: 12px 11px; text-align: center;
                box-shadow: 0 8px 22px rgba(120,90,10,.30); }
  .champ-card .trophy { font-size: 22px; }
  .champ-card .cname { font-size: 16px; font-weight: 800; color: #fde68a; margin-top: 2px;
                       white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .champ-card .cwin { font-size: 11px; font-weight: 800; color: #fbbf24; margin-top: 1px; }
  .champ-card .crun { font-size: 9.5px; font-weight: 700; color: #94a3b8; margin-top: 3px; }
  .tbd { color: #475569; font-weight: 700; font-style: italic; }
</style>
"""

_JS = """
<script>
(function () {
  var NEXT = {R32:'R16', R16:'QF', QF:'SF', SF:'Final', Final:'Champion'};
  function fit() {
    try {
      var h = document.body.scrollHeight;
      if (window.frameElement) { window.frameElement.style.height = (h + 4) + 'px'; }
    } catch (e) {}
  }
  function draw() {
    try {
      var wrap = document.querySelector('.bracket');
      var svg = document.querySelector('svg.links');
      if (!wrap || !svg) return;
      var W = wrap.scrollWidth, H = wrap.scrollHeight;
      svg.setAttribute('width', W); svg.setAttribute('height', H);
      svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      var base = wrap.getBoundingClientRect();
      function box(el) {
        var r = el.getBoundingClientRect();
        return { l: r.left - base.left + wrap.scrollLeft, r: r.right - base.left + wrap.scrollLeft,
                 m: (r.top + r.bottom) / 2 - base.top + wrap.scrollTop };
      }
      document.querySelectorAll('.match').forEach(function (el) {
        var rd = el.dataset.round, idx = parseInt(el.dataset.idx, 10);
        var parent = document.getElementById('m-' + NEXT[rd] + '-' + Math.floor(idx / 2));
        if (!parent) return;
        var c = box(el), p = box(parent);
        var midx = (c.r + p.l) / 2;
        var d = 'M' + c.r + ',' + c.m + ' H' + midx + ' V' + p.m + ' H' + p.l;
        var path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        path.setAttribute('d', d); path.setAttribute('fill', 'none');
        path.setAttribute('stroke', 'rgba(148,163,184,.34)'); path.setAttribute('stroke-width', '1.5');
        svg.appendChild(path);
      });
    } catch (e) {}
  }
  function refresh() { draw(); fit(); }
  refresh();
  window.addEventListener('load', refresh);
  window.addEventListener('resize', refresh);
  setTimeout(refresh, 80); setTimeout(refresh, 320);
})();
</script>
"""


def _esc(text) -> str:
    return html.escape(str(text)) if text is not None else ""


def _flag_html(team: str) -> str:
    emoji = flags.flag(team)
    if emoji:
        return f'<span class="flag">{emoji}</span>'
    return f'<span class="code">{_esc(flags.code(team))}</span>'


def _slot_html(slot_list: list, fav: bool, label: str | None) -> str:
    """One participant slot: flag + (short) name + 'reach this slot' %, with a faint runner-up."""
    if not slot_list:
        body = '<span class="flag">·</span><span class="name tbd">TBD</span>'
        return f'<div class="slot dim">{body}</div>'
    d0 = slot_list[0]
    team = d0["team"]
    sec = ""
    if len(slot_list) > 1 and d0["p"] < 0.80:
        d1 = slot_list[1]
        sec = f'<span class="sec">· {_esc(flags.code(d1["team"]))} {d1["p"] * 100:.0f}%</span>'
    cls = "slot fav" if fav else "slot dim"
    inner = (
        f"{_flag_html(team)}"
        f'<span class="name" title="{_esc(team)}">{_esc(flags.short_name(team))}{sec}</span>'
        f'<span class="pct">{d0["p"] * 100:.0f}%</span>'
    )
    out = f'<div class="{cls}">{inner}</div>'
    if label:
        out += f'<div class="slotlabel">{_esc(label)}</div>'
    return out


def _split_html(match: dict) -> str:
    """Simulated winner split: each modal participant's *unconditional* chance to win this match.

    The two segments are the modal top/bottom teams' advancement probabilities (P they win this
    match node); a neutral remainder fills the rest. When the participants are settled the remainder
    vanishes and it reads as a clean head-to-head; when a slot is still wide open (e.g. a Round-of-32
    third-place slot) the grey gap honestly shows the opponent isn't decided — instead of inflating
    the favourite to a misleading 100%.
    """
    top = match["top"][0] if match["top"] else None
    bot = match["bottom"][0] if match["bottom"] else None
    if not top or not bot:
        return '<div class="mid"><div class="splitbar"></div></div>'
    adv = {d["team"]: d["p"] for d in match["advance"]}
    a = max(0.0, adv.get(top["team"], 0.0)) * 100.0
    b = max(0.0, adv.get(bot["team"], 0.0)) * 100.0
    if a + b > 100.0:  # numerical safety only
        a, b = 100.0 * a / (a + b), 100.0 * b / (a + b)
    rest = max(0.0, 100.0 - a - b)
    return (
        '<div class="mid"><div class="splitbar">'
        f'<div class="seg" style="width:{a:.1f}%;background:{_TOP}"></div>'
        f'<div class="seg rest" style="width:{rest:.1f}%"></div>'
        f'<div class="seg" style="width:{b:.1f}%;background:{_BOT}"></div>'
        "</div></div>"
        f'<div class="winpct"><span class="l">{a:.0f}%</span>'
        f'<span class="r">{b:.0f}%</span></div>'
    )


def _played_for(match: dict, played: dict | None):
    """If both modal participants formed an already-played tie, return its oriented score line."""
    if not played or not match["top"] or not match["bottom"]:
        return None
    a, b = match["top"][0]["team"], match["bottom"][0]["team"]
    rec = played.get(frozenset((a, b)))
    if not rec:
        return None
    sa = rec["score"].get(a)
    sb = rec["score"].get(b)
    if sa is None or sb is None:
        return None
    return a, sa, b, sb


def _marks(rk: str, m: int, paths: dict, toss: dict | None) -> dict:
    """Which highlights touch match (rk, m): the traced sides per kind + a toss-up flag."""
    kinds: dict[str, str] = {}  # kind -> side ("top"/"bottom") it traces here
    for kind in ("champ", "upset", "sel"):
        hit = paths.get(kind, {}).get(rk)
        if hit is not None and hit["index"] == m:
            kinds[kind] = hit["side"]
    is_toss = bool(toss and toss["round"] == rk and toss["index"] == m)
    return {"kinds": kinds, "toss": is_toss}


def _match_html(match: dict, mark: dict, played: dict | None) -> str:
    rk, m = match["round"], match["index"]
    advance = match["advance"]
    fav_team = advance[0]["team"] if advance else None
    top_fav = bool(match["top"] and match["top"][0]["team"] == fav_team)
    bot_fav = bool(match["bottom"] and match["bottom"][0]["team"] == fav_team)

    # Border accent by priority + a row of dots for any extra memberships.
    members = list(mark["kinds"].keys()) + (["toss"] if mark["toss"] else [])
    primary = next((k for k in _PRIORITY if k in members), None)
    style = f"border-left-color:{_ACCENT[primary]}" if primary else ""
    dots = "".join(
        f'<span class="dot" style="background:{_ACCENT[k]}"></span>'
        for k in members
        if k != primary
    )
    tag = ""
    if mark["toss"]:
        tag = f'<span class="tag" style="background:{_ACCENT["toss"]};color:#3a2606">toss-up</span>'

    played_line = _played_for(match, played)
    if played_line is not None:
        a, sa, b, sb = played_line
        mid = (
            f'<div class="played">{_esc(flags.short_name(a))} {sa}–{sb} '
            f'{_esc(flags.short_name(b))} <span class="ftbadge">FT</span></div>'
        )
    else:
        mid = _split_html(match)

    return (
        f'<div class="match" id="m-{rk}-{m}" data-round="{rk}" data-idx="{m}" style="{style}">'
        f'<div class="tags">{tag}{dots}</div>'
        f"{_slot_html(match['top'], top_fav, match.get('top_label'))}"
        f"{mid}"
        f"{_slot_html(match['bottom'], bot_fav, match.get('bottom_label'))}"
        "</div>"
    )


def _band(content: str, mult: int) -> str:
    return f'<div class="band" style="height:calc(var(--band) * {mult})">{content}</div>'


def _champion_html(bracket: dict) -> str:
    champ = bracket.get("champion") or []
    if not champ:
        body = '<div class="trophy">🏆</div><div class="cname tbd">TBD</div>'
    else:
        c0 = champ[0]
        runner = (
            f'<div class="crun">then {_esc(flags.short_name(champ[1]["team"]))} '
            f"{champ[1]['p'] * 100:.0f}%</div>"
            if len(champ) > 1
            else ""
        )
        body = (
            '<div class="trophy">🏆</div>'
            f'<div class="cname" title="{_esc(c0["team"])}">{_flag_html(c0["team"])} '
            f"{_esc(flags.short_name(c0['team']))}</div>"
            f'<div class="cwin">{c0["p"] * 100:.0f}% to lift the trophy</div>'
            f"{runner}"
        )
    head = '<div class="col-head">Champion<span class="n">most likely</span></div>'
    inner = f'<div class="champ-card" id="m-Champion-0">{body}</div>'
    return f'<div class="col">{head}<div class="col-body">{_band(inner, 16)}</div></div>'


def _column_html(rnd: dict, paths: dict, toss: dict | None, played: dict | None) -> str:
    rk = rnd["key"]
    mult = _BAND_MULT[rk]
    bands = "".join(
        _band(_match_html(match, _marks(rk, match["index"], paths, toss), played), mult)
        for match in rnd["matches"]
    )
    head = (
        f'<div class="col-head">{_esc(rnd["title"])}'
        f'<span class="n">{len(rnd["matches"])} match{"es" if len(rnd["matches"]) != 1 else ""}'
        "</span></div>"
    )
    return f'<div class="col">{head}<div class="col-body">{bands}</div></div>'


def _chip(team: str) -> str:
    """Inline 'flag + name' for a native-Streamlit highlight card."""
    f = flags.flag(team)
    return f"{f} {team}" if f else f"{flags.code(team)} · {team}"


def render_highlights(highlights: dict) -> None:
    """Three native call-out cards above the bracket: champion, Cinderella, closest call."""
    champ, upset, toss = (
        highlights.get("champion"),
        highlights.get("upset"),
        highlights.get("toss_up"),
    )
    c1, c2, c3 = st.columns(3)
    with c1, st.container(border=True):
        st.markdown("👑 **Projected champion**")
        if champ:
            st.markdown(f"#### {_chip(champ['team'])}")
            st.caption(f"{champ['p'] * 100:.1f}% to lift the trophy across the simulations")
        else:
            st.caption("No clear favourite yet.")
    with c2, st.container(border=True):
        st.markdown("✨ **Cinderella watch**")
        if upset:
            st.markdown(f"#### {_chip(upset['team'])}")
            st.caption(
                f"{upset['p'] * 100:.0f}% to reach the {upset['stage_title']} — "
                f"only the #{upset['elo_rank']} side by Elo"
            )
        else:
            st.caption("Favourites dominate — no standout underdog run.")
    with c3, st.container(border=True):
        st.markdown("🎲 **Closest call**")
        if toss:
            st.markdown(f"#### {flags.code(toss['team_a'])} v {flags.code(toss['team_b'])}")
            st.caption(
                f"{toss['round_title']}: {toss['p_a'] * 100:.0f}% / {toss['p_b'] * 100:.0f}% — "
                "the bracket's tightest projected tie"
            )
        else:
            st.caption("No toss-up among the determined matchups.")


def render_legend(selected_team: str | None = None) -> None:
    """A compact colour key for the four highlighted paths (matches the bracket accents)."""
    items = [
        (_ACCENT["champ"], "Champion path"),
        (_ACCENT["upset"], "Cinderella path"),
        (_ACCENT["toss"], "Toss-up"),
        (_ACCENT["sel"], f"Your team{f': {selected_team}' if selected_team else ''}"),
    ]
    swatches = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:5px;margin-right:14px">'
        f'<span style="width:11px;height:11px;border-radius:3px;background:{c};'
        f'display:inline-block"></span>'
        f'<span style="font-size:12.5px;color:#94a3b8">{_esc(lbl)}</span></span>'
        for c, lbl in items
    )
    st.markdown(f'<div style="padding:2px 0 4px">{swatches}</div>', unsafe_allow_html=True)


def render_bracket(
    result,
    ratings: dict | None = None,
    *,
    selected_team: str | None = None,
    played: dict | None = None,
    highlights: dict | None = None,
) -> None:
    """Render the full knockout bracket for a finished simulation into the Streamlit page.

    ``result`` is a :class:`core.simulate.SimulationResult` (its ``bracket`` projection is required);
    ``ratings`` (team→Elo) powers the Cinderella highlight; ``selected_team`` traces a chosen team's
    road; ``played`` maps ``frozenset({a, b}) → {"score": {team: goals}}`` for any already-played tie.
    """
    bracket = getattr(result, "bracket", None)
    if not bracket:
        st.info("Run a simulation to draw the bracket.")
        return

    highlights = highlights or cb.derive_highlights(result, ratings)
    champ_team = highlights["champion"]["team"] if highlights["champion"] else None
    upset_team = highlights["upset"]["team"] if highlights["upset"] else None
    paths = {
        "champ": cb.team_path(bracket, champ_team),
        "upset": cb.team_path(bracket, upset_team),
        "sel": cb.team_path(bracket, selected_team),
    }
    toss = highlights["toss_up"]

    columns = "".join(_column_html(rnd, paths, toss, played) for rnd in bracket["rounds"])
    columns += _champion_html(bracket)
    col_h = 16 * _BAND_PX
    body = (
        f'<div class="wrap" style="--band:{_BAND_PX}px">'
        '<div class="scroll"><div class="bracket">'
        '<svg class="links" xmlns="http://www.w3.org/2000/svg"></svg>'
        f"{columns}"
        "</div></div></div>"
    )
    # Initial height fits the funnel; the JS pass trims it to the exact content height once laid out.
    components.html(_CSS + body + _JS, height=col_h + 150, scrolling=False)
