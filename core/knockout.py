"""Official FIFA World Cup 2026 knockout progression — the bracket DAG + Annexe C, loaded from the
committed rules file (``data/fifa_world_cup_2026_rules.json``) as the single source of truth.

This module is the authoritative, framework-free description of *how the 32 knockout qualifiers are
slotted and play down to the champion*. It exists so the Monte-Carlo simulator
(:mod:`core.simulate`) and its bracket projection (:mod:`core.bracket`) implement the real 2026
rules rather than an invented template:

* the **match DAG** — Round-of-32 (M73) → Final (M104) — with each match's two participant slot
  tokens and the token its winner advances as, in dependency order; and
* **Annexe C** — FIFA's 495-row table that, given *which eight groups* supplied the best
  third-placed qualifiers, assigns each of the eight Annexe-eligible group winners its third-place
  opponent (the ranking decides *who* qualifies, never *whom* they face).

Token taxonomy the simulator's resolver handles (see the rules file ``notation`` block):

* ``1X`` / ``2X`` / ``3X`` — winner / runner-up / third-placed team of group ``X``;
* ``ANNEX_C_FOR_1X`` — the best-third opponent assigned to group-``X`` winner by Annexe C;
* ``W{n}`` / ``L{n}`` — winner / loser of match ``M{n}``.

Pure (json + dataclasses only): no numpy, no Streamlit. Unit-tested in ``tests/test_knockout.py``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from core import config

# Prefix marking a Round-of-32 slot routed to a best third-placed team via Annexe C.
ANNEX_PREFIX = "ANNEX_C_FOR_"

# The five rounds that form the advancement tree to the champion. The third-place match (M103) is a
# side branch — it has no bearing on who reaches each stage — so it is excluded from this geometry.
# Ordered shallow → deep.
ADVANCEMENT_ROUNDS = ["R32", "R16", "QF", "SF", "Final"]


@dataclass(frozen=True)
class Match:
    """One knockout match: raw slot tokens for its two participants + how its result is referenced."""

    id: str  # "M73" … "M104"
    round: str  # "R32" / "R16" / "QF" / "SF" / "3P" / "Final"
    a: str  # raw slot token for team_a (e.g. "2A", "1E", "ANNEX_C_FOR_1E", "W74")
    b: str  # raw slot token for team_b
    win_token: str | None  # token the winner advances as ("W73"); None for the 3P match & the final
    # token the loser is referenced as ("L101"); only the semis' losers (M101/M102) are consumed.
    lose_token: str


@dataclass(frozen=True)
class KnockoutSpec:
    """The whole knockout phase: ordered matches + the Annexe C matrix, read from the rules file."""

    matches: tuple[Match, ...]  # dependency order M73 → M104 (includes the 3P match)
    annex_matrix: dict[str, dict[str, str]]  # 8-group key → {"1A": "3E", …}
    winner_slot_to_match: dict[str, str]  # "1A" → "M79" (Annexe-eligible winners only)
    final_id: str  # the match whose winner is the champion ("M104")


@dataclass(frozen=True)
class Layout:
    """Occupancy geometry: how matches map onto the perfect-binary-tree participant slots.

    ``match_row`` maps each advancement-tree match to ``(round_key, row)``; ``rows_by_round`` is the
    inverse (round → match ids ordered by row); ``r32_slot_order`` lists the 32 Round-of-32
    participant slots in bracket order as ``(match_id, side)`` with ``side`` ∈ {"a", "b"}.
    """

    match_row: dict[str, tuple[str, int]]
    rows_by_round: dict[str, list[str]]
    r32_slot_order: list[tuple[str, str]]


def load_rules(path: Path | None = None) -> dict:
    """Load the committed 2026 progression-rules JSON (bracket DAG + Annexe C matrix)."""
    path = path or config.WC2026_RULES_PATH
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def build_spec(rules: dict) -> KnockoutSpec:
    """Flatten the rules file's ``bracket`` into an ordered match list + the Annexe C matrix.

    Matches are emitted Round-of-32 → Final, which is also a valid dependency order (every ``W``/``L``
    a match references is produced earlier in the list), so the simulator can resolve and play them
    in a single forward pass.
    """
    bracket = rules["bracket"]
    matches: list[Match] = []

    def add(entry: dict, round_key: str) -> None:
        mid = entry["match_id"]
        matches.append(
            Match(
                id=mid,
                round=round_key,
                a=entry["team_a"]["slot"],
                b=entry["team_b"]["slot"],
                win_token=entry.get("winner_advances_as"),
                lose_token="L" + mid[1:],
            )
        )

    for entry in bracket["round_of_32"]:
        add(entry, "R32")
    for entry in bracket["round_of_16"]:
        add(entry, "R16")
    for entry in bracket["quarter_finals"]:
        add(entry, "QF")
    for entry in bracket["semi_finals"]:
        add(entry, "SF")
    add(bracket["third_place_match"], "3P")
    add(bracket["final"], "Final")

    alloc = rules["best_third_place_allocation"]
    annex_matrix = {
        key: dict(row["assignments_by_group_winner"]) for key, row in alloc["matrix"].items()
    }
    return KnockoutSpec(
        matches=tuple(matches),
        annex_matrix=annex_matrix,
        winner_slot_to_match=dict(alloc["winner_slot_to_round_of_32_match"]),
        final_id=bracket["final"]["match_id"],
    )


def annex_assignment(annex_matrix: dict, qualifying_third_groups) -> dict[str, str]:
    """Annexe C lookup: the third-place opponent each eligible group winner faces.

    ``qualifying_third_groups`` is the set/iterable of the eight group letters that supplied the best
    third-placed qualifiers. The key is those letters sorted and concatenated (e.g. ``"ABCDEFGH"``);
    the returned dict maps each Annexe-eligible winner slot (``"1A"`` …) to a third-place slot
    (``"3E"`` …). The matrix guarantees no team meets its own group's third (``1X`` is never paired
    with ``3X``).
    """
    key = "".join(sorted(qualifying_third_groups))
    return annex_matrix[key]


def occupancy_layout(spec: KnockoutSpec) -> Layout:
    """Derive the perfect-binary-tree row of every advancement-tree match.

    The geometry restores the identity :mod:`core.bracket` relies on — *match m of a round owns
    participant slots 2m / 2m+1, and its winner is slot m of the next round*. The official 2026
    bracket crosses at **every** round (it is not only the R32→R16 feed that is non-sequential — the
    quarter-finals pair W89·W90 / W93·W94 / W91·W92 / W95·W96, for instance), so the rows are
    assigned top-down from the single Final match: a match placed at row ``r`` forces the two matches
    that feed it to rows ``2r`` and ``2r+1``. Walking Final → SF → QF → R16 lays every row down
    consistently, no per-round special-casing.
    """
    by_round = {rk: [m for m in spec.matches if m.round == rk] for rk in ADVANCEMENT_ROUNDS}
    produces = {m.win_token: m for m in spec.matches if m.win_token}

    row: dict[str, int] = {by_round["Final"][0].id: 0}
    for rk in ["Final", "SF", "QF", "R16"]:  # assign the rows of each round's two feeder matches
        for m in by_round[rk]:
            r = row[m.id]
            row[produces[m.a].id] = 2 * r
            row[produces[m.b].id] = 2 * r + 1

    match_row = {m.id: (m.round, row[m.id]) for rk in ADVANCEMENT_ROUNDS for m in by_round[rk]}
    rows_by_round: dict[str, list[str]] = {
        rk: [""] * len(by_round[rk]) for rk in ADVANCEMENT_ROUNDS
    }
    for rk in ADVANCEMENT_ROUNDS:
        for m in by_round[rk]:
            rows_by_round[rk][row[m.id]] = m.id

    r32_slot_order: list[tuple[str, str]] = []
    for mid in rows_by_round["R32"]:
        r32_slot_order.append((mid, "a"))
        r32_slot_order.append((mid, "b"))

    return Layout(match_row=match_row, rows_by_round=rows_by_round, r32_slot_order=r32_slot_order)
