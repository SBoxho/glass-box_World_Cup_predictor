"""Guardrail — team → flag emoji / ISO fallback is correct and total (never raises).

:mod:`core.flags` is a tiny lookup the bracket leans on for every card, so its three contracts are
pinned: sovereign nations build a regional-indicator emoji, the UK home nations use subdivision tag
sequences, and *every* name (even unknown ones) yields a printable short code so a card always has a
badge to show.
"""

from __future__ import annotations

from core import flags


def test_sovereign_flag_is_regional_indicator_pair():
    # Brazil → 🇧🇷 = REGIONAL INDICATOR B + R.
    assert flags.flag("Brazil") == "\U0001f1e7\U0001f1f7"
    assert flags.iso2("Brazil") == "BR"
    assert flags.flag("United States") == "\U0001f1fa\U0001f1f8"


def test_home_nations_use_subdivision_tag_sequences():
    # England → 🏴 + tag('gbeng') + cancel tag. No alpha-2 of its own.
    assert flags.flag("England") == (
        "\U0001f3f4\U000e0067\U000e0062\U000e0065\U000e006e\U000e0067\U000e007f"
    )
    assert flags.iso2("England") is None  # not a sovereign alpha-2
    assert flags.code("England") == "ENG"
    assert flags.code("Scotland") == "SCO"


def test_code_is_total_even_for_unknown_names():
    # Unknown name: no flag, but still a printable 3-letter badge (the ISO-based fallback).
    assert flags.flag("Wakanda") == ""
    assert flags.iso2("Wakanda") is None
    assert flags.code("Wakanda") == "WAK"
    assert flags.code("") == "?"
    assert flags.code(None) == "?"


def test_short_name_abbreviates_only_the_long_ones():
    assert flags.short_name("United States") == "USA"
    assert flags.short_name("Bosnia and Herzegovina") == "Bosnia & Herz."
    assert flags.short_name("Brazil") == "Brazil"  # already short → unchanged


def test_every_2026_team_has_a_flag_and_code():
    # The 48-team field must all render: a flag emoji (or home-nation tag) and a short code.
    from core.ingest import load_wc2026

    wc = load_wc2026()
    teams = {t for grp in wc["groups"].values() for t in grp}
    for t in teams:
        assert flags.flag(t), f"no flag for {t}"
        assert flags.code(t) and flags.code(t) != "?", f"no code for {t}"
