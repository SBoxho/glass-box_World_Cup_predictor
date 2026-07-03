"""Guardrail — the bracket card renders an already-played *knockout* tie honestly.

The knockout bracket cards stamp a score on ties whose two most-likely teams have actually met. A
tie decided in extra time or on penalties must not read as the draw its full-time line alone would
suggest (e.g. "NED 1–1 MAR" for a shootout Morocco won): the card sources the winner + how the tie
was decided from ``known_ko_results`` (``components.bracket._played_ko_line`` / ``_match_html``) and
shows the real winner, the shootout tally, and dims the eliminated side. These tests pin that
derivation/rendering; the ft-only group stamp must keep working unchanged.

``components`` lives under ``app/`` (added to ``sys.path`` at runtime by ``streamlit_app``), so the
test puts it on the path the same way before importing the presentation module.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from components import bracket as bv  # noqa: E402  (needs the app/ path insert above)

# A card mark with no highlights/toss-up — the rendering under test is independent of the accents.
_MARK = {"kinds": {}, "toss": False}


def _match(top: list, bottom: list, advance: list, rk: str = "R32", idx: int = 0) -> dict:
    return {"round": rk, "index": idx, "top": top, "bottom": bottom, "advance": advance}


def _one(team: str) -> list:
    return [{"team": team, "p": 1.0}]


# --------------------------------------------------------------------------------------
# _played_ko_line — orientation, badge, winner
# --------------------------------------------------------------------------------------
def test_played_ko_line_penalty_orients_and_badges():
    match = _match(_one("Netherlands"), _one("Morocco"), [])
    ko = {
        frozenset(("Netherlands", "Morocco")): {
            "home": "Netherlands",
            "away": "Morocco",
            "home_score": 1,
            "away_score": 1,
            "winner": "Morocco",
            "decided_by": "pen",
            "pens": [2, 3],
        }
    }
    assert bv._played_ko_line(match, ko) == ("Netherlands", 1, "Morocco", 1, "2–3 pens", "Morocco")


def test_played_ko_line_flips_a_reversed_record_to_card_order():
    # The stored record is oriented (home=Morocco), but the card lists Netherlands on top — scores
    # and the shootout tally must both flip so the line reads in the card's own top/bottom order.
    match = _match(_one("Netherlands"), _one("Morocco"), [])
    ko = {
        frozenset(("Netherlands", "Morocco")): {
            "home": "Morocco",
            "away": "Netherlands",
            "home_score": 1,
            "away_score": 1,
            "winner": "Morocco",
            "decided_by": "pen",
            "pens": [3, 2],
        }
    }
    assert bv._played_ko_line(match, ko) == ("Netherlands", 1, "Morocco", 1, "2–3 pens", "Morocco")


def test_played_ko_line_extra_time_uses_aet_badge():
    match = _match(_one("Spain"), _one("Italy"), [])
    ko = {
        frozenset(("Spain", "Italy")): {
            "home": "Spain",
            "away": "Italy",
            "home_score": 2,
            "away_score": 1,
            "winner": "Spain",
            "decided_by": "et",
        }
    }
    assert bv._played_ko_line(match, ko) == ("Spain", 2, "Italy", 1, "a.e.t.", "Spain")


def test_played_ko_line_none_when_no_record_or_scores_missing():
    match = _match(_one("Spain"), _one("Italy"), [])
    assert bv._played_ko_line(match, None) is None
    assert bv._played_ko_line(match, {}) is None
    # Decided but no goals in the feed -> no line (the simulated split renders instead).
    ko = {
        frozenset(("Spain", "Italy")): {
            "home": "Spain",
            "away": "Italy",
            "home_score": None,
            "away_score": None,
            "winner": "Spain",
            "decided_by": "pen",
            "pens": [4, 2],
        }
    }
    assert bv._played_ko_line(match, ko) is None


# --------------------------------------------------------------------------------------
# _match_html — the rendered card
# --------------------------------------------------------------------------------------
def test_match_html_penalty_tie_highlights_winner_and_shootout():
    match = _match(_one("Netherlands"), _one("Morocco"), _one("Morocco"))
    ko = {
        frozenset(("Netherlands", "Morocco")): {
            "home": "Netherlands",
            "away": "Morocco",
            "home_score": 1,
            "away_score": 1,
            "winner": "Morocco",
            "decided_by": "pen",
            "pens": [2, 3],
        }
    }
    html = bv._match_html(match, _MARK, played=None, ko_played=ko)
    assert "1–1" in html
    assert '<span class="ftbadge">2–3 pens</span>' in html
    assert '<span class="win">Morocco</span>' in html  # advancing side brightened
    assert '<span class="lose">Netherlands</span>' in html  # eliminated side dimmed
    assert "FT" not in html  # not stamped as a plain full-time draw


def test_match_html_ko_takes_precedence_over_ft_only_map():
    match = _match(_one("Netherlands"), _one("Morocco"), _one("Morocco"))
    # The ft-only ``played`` map would show a bare "1–1 FT"; the richer ko record must win.
    played = {frozenset(("Netherlands", "Morocco")): {"score": {"Netherlands": 1, "Morocco": 1}}}
    ko = {
        frozenset(("Netherlands", "Morocco")): {
            "home": "Netherlands",
            "away": "Morocco",
            "home_score": 1,
            "away_score": 1,
            "winner": "Morocco",
            "decided_by": "pen",
            "pens": [2, 4],
        }
    }
    html = bv._match_html(match, _MARK, played=played, ko_played=ko)
    assert "2–4 pens" in html
    assert '<span class="win">Morocco</span>' in html


def test_match_html_group_ft_stamp_unchanged():
    # A group result (ft-only map, no ko record) still renders the plain "FT" stamp with no
    # winner/loser emphasis — the change must not regress the group-stage cards.
    match = _match(_one("Brazil"), _one("Serbia"), _one("Brazil"))
    played = {frozenset(("Brazil", "Serbia")): {"score": {"Brazil": 2, "Serbia": 0}}}
    html = bv._match_html(match, _MARK, played=played, ko_played=None)
    assert "2–0" in html
    assert '<span class="ftbadge">FT</span>' in html
    assert 'class="win"' not in html
    assert 'class="lose"' not in html


def test_match_html_no_played_data_falls_back_to_simulated_split():
    match = _match(_one("Brazil"), _one("Serbia"), _one("Brazil"))
    html = bv._match_html(match, _MARK, played=None, ko_played=None)
    assert 'class="played"' not in html  # no score line
    assert 'class="splitbar"' in html  # the simulated winner split instead
