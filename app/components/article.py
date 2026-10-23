"""The mid-tournament Medium article — a dated, public record of what the model said.

Shared by the Matchday Home page and the Simulate tab's pre-tournament replay, so both quote the
same numbers. Those numbers are read from ``reports/sim_1M.json`` (the 1,000,000-run simulation the
article was written from), never typed in by hand.
"""

from __future__ import annotations

import json

import streamlit as st

from core import config

ARTICLE_URL = (
    "https://medium.com/@SebastienB./i-simulated-the-fifa-2026-world-cup-1-million-times-here-is-"
    "what-the-model-predicts-bf68aa9d23fb"
)
ARTICLE_TITLE = "I simulated the FIFA 2026 World Cup 1 million times"


@st.cache_data(show_spinner=False)
def load_article_snapshot():
    """Headline numbers of the 1M-run simulation behind the article, or ``None`` if unavailable."""
    try:
        return json.loads((config.BASE_DIR / "reports" / "sim_1M.json").read_text(encoding="utf-8"))
    except Exception:
        return None


def article_markdown() -> str:
    """One markdown line: the article link + the 1M-run numbers it was written from."""
    snap = load_article_snapshot() or {}
    line = f"📝 **Written during the tournament:** [{ARTICLE_TITLE}]({ARTICLE_URL})"
    meta = snap.get("meta", {})
    top = snap.get("champion_ranking", [])[:2]
    fin = snap.get("most_likely_final", {})
    n_locked = meta.get("n_locked_group_results")
    if top and n_locked is not None:
        odds = ", ".join(f"**{t['team']} {t['p_champion']:.1%}**" for t in top)
        line += (
            f" — a {meta.get('n_sims', 0):,}-run simulation on "
            f"{str(meta.get('live_fetched_at', ''))[:10]} ({n_locked} of 72 group games played, no "
            f"knockout game yet) had {odds} as the top two, and "
            f"**{' vs '.join(fin.get('teams', []))}** as the most likely final."
        )
    return line
