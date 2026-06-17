"""Team → flag emoji / ISO country code, with a graceful ISO-based fallback.

A tiny, framework-free lookup so the UI can show a flag next to a national team. Names are the
canonical spellings produced by :func:`core.config.normalize_team` (the same ones used in the draw
and the Elo history), so this joins cleanly to everything else in the project.

* :func:`flag` returns a Unicode flag emoji built from the ISO-3166-1 alpha-2 code via regional
  indicator symbols (e.g. ``"BR"`` → 🇧🇷). The four UK home nations have no alpha-2 of their own,
  so England/Scotland/Wales use the special subdivision *tag* sequences (🏴󠁧󠁢󠁥󠁮󠁧󠁿 …).
* :func:`code` returns a short uppercase label (alpha-2, or ``ENG``/``SCO``/… for the home nations)
  — the **ISO-based fallback** a renderer can show when emoji are unavailable (e.g. some Windows
  builds render flag emoji as the bare letters anyway).
* :func:`short_name` shortens a few long names so they fit a compact bracket card.

Pure and dependency-free (only the stdlib), so it is unit-tested directly and importable from both
``app`` and ``api`` without pulling in Streamlit.
"""

from __future__ import annotations

# Canonical team name -> ISO-3166-1 alpha-2. Covers the 48-team 2026 field plus a margin of other
# nations that show up in the Elo history / predictor, so a flag is available app-wide. Unknown
# names simply have no flag (the renderer falls back to a neutral badge).
ISO2: dict[str, str] = {
    # --- 2026 World Cup field ---------------------------------------------------------------
    "Mexico": "MX",
    "South Africa": "ZA",
    "South Korea": "KR",
    "Czechia": "CZ",
    "Canada": "CA",
    "Bosnia and Herzegovina": "BA",
    "Qatar": "QA",
    "Switzerland": "CH",
    "Brazil": "BR",
    "Morocco": "MA",
    "Haiti": "HT",
    "United States": "US",
    "Paraguay": "PY",
    "Australia": "AU",
    "Turkey": "TR",
    "Germany": "DE",
    "Curaçao": "CW",
    "Ivory Coast": "CI",
    "Ecuador": "EC",
    "Netherlands": "NL",
    "Japan": "JP",
    "Sweden": "SE",
    "Tunisia": "TN",
    "Belgium": "BE",
    "Egypt": "EG",
    "Iran": "IR",
    "New Zealand": "NZ",
    "Spain": "ES",
    "Cape Verde": "CV",
    "Saudi Arabia": "SA",
    "Uruguay": "UY",
    "France": "FR",
    "Senegal": "SN",
    "Iraq": "IQ",
    "Norway": "NO",
    "Argentina": "AR",
    "Algeria": "DZ",
    "Austria": "AT",
    "Jordan": "JO",
    "Portugal": "PT",
    "DR Congo": "CD",
    "Uzbekistan": "UZ",
    "Colombia": "CO",
    "Croatia": "HR",
    "Ghana": "GH",
    "Panama": "PA",
    # --- other nations commonly seen in the Elo history / predictor -------------------------
    "Italy": "IT",
    "Poland": "PL",
    "Serbia": "RS",
    "Denmark": "DK",
    "Nigeria": "NG",
    "Cameroon": "CM",
    "Ukraine": "UA",
    "Russia": "RU",
    "Peru": "PE",
    "Chile": "CL",
    "Costa Rica": "CR",
    "Honduras": "HN",
    "Venezuela": "VE",
    "Bolivia": "BO",
    "Mali": "ML",
    "Greece": "GR",
    "Hungary": "HU",
    "Romania": "RO",
    "Slovakia": "SK",
    "Slovenia": "SI",
    "China": "CN",
    "North Korea": "KP",
    "Taiwan": "TW",
    "Republic of Ireland": "IE",
    "Ireland": "IE",
}

# UK home nations have no alpha-2 of their own — they use ISO-3166-2 subdivision codes, rendered as
# special flag *tag sequences*. (Used by both :func:`flag` and :func:`code`.)
_SUBDIVISIONS: dict[str, tuple[str, str]] = {
    # team -> (subdivision tag for the emoji, short label for the text fallback)
    "England": ("gbeng", "ENG"),
    "Scotland": ("gbsct", "SCO"),
    "Wales": ("gbwls", "WAL"),
    "Northern Ireland": ("gbnir", "NIR"),
}

_REGIONAL_INDICATOR_A = 0x1F1E6  # 🇦 — alpha-2 letters map to these regional-indicator symbols
_TAG_BASE = 0xE0000  # tag latin letters used in subdivision flag sequences
_BLACK_FLAG = "\U0001f3f4"  # 🏴 — base of a subdivision flag sequence
_CANCEL_TAG = "\U000e007f"  # terminates a tag sequence


def _regional_indicators(alpha2: str) -> str:
    """``"BR"`` → 🇧🇷 (two regional-indicator symbols)."""
    return "".join(chr(_REGIONAL_INDICATOR_A + (ord(c) - ord("A"))) for c in alpha2.upper())


def _subdivision_flag(tag: str) -> str:
    """Build a UK-home-nation flag (``"gbeng"`` → 🏴󠁧󠁢󠁥󠁮󠁧󠁿) from its subdivision tag."""
    return _BLACK_FLAG + "".join(chr(_TAG_BASE + ord(c)) for c in tag) + _CANCEL_TAG


def iso2(team: str | None) -> str | None:
    """ISO-3166-1 alpha-2 for a canonical team name, or ``None`` if unmapped / a UK home nation."""
    return ISO2.get((team or "").strip())


def flag(team: str | None) -> str:
    """Unicode flag emoji for a team, or ``""`` when no flag is known.

    Sovereign nations use regional-indicator pairs; the UK home nations use subdivision tag
    sequences. An unknown name returns the empty string so the caller can drop in a text badge.
    """
    name = (team or "").strip()
    if name in _SUBDIVISIONS:
        return _subdivision_flag(_SUBDIVISIONS[name][0])
    a2 = ISO2.get(name)
    return _regional_indicators(a2) if a2 else ""


def code(team: str | None) -> str:
    """Short uppercase label (the ISO-based fallback): alpha-2, a home-nation tag, or initials.

    Always returns *something* printable so a flag-less renderer still shows a recognizable badge:
    the alpha-2 code where known, ``ENG``/``SCO``/… for the home nations, else the first three
    letters of the name upper-cased.
    """
    name = (team or "").strip()
    if name in _SUBDIVISIONS:
        return _SUBDIVISIONS[name][1]
    a2 = ISO2.get(name)
    if a2:
        return a2
    return name[:3].upper() if name else "?"


# A few names are too long for a compact bracket card; shorten just those, pass everything else
# through unchanged (the full name stays available as a tooltip in the renderer).
_SHORT_NAMES: dict[str, str] = {
    "Bosnia and Herzegovina": "Bosnia & Herz.",
    "United States": "USA",
    "Saudi Arabia": "Saudi Arabia",
    "South Africa": "South Africa",
    "South Korea": "South Korea",
    "New Zealand": "New Zealand",
    "Republic of Ireland": "Ireland",
    "Northern Ireland": "N. Ireland",
}


def short_name(team: str | None) -> str:
    """A display name short enough for a bracket card (long names abbreviated, others unchanged)."""
    name = (team or "").strip()
    return _SHORT_NAMES.get(name, name)
