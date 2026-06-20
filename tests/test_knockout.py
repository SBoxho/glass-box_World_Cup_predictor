"""Guardrail — the official 2026 knockout DAG + Annexe C load and lay out correctly.

The knockout structure and the 495-row third-place matrix are loaded from the committed rules file
(``data/fifa_world_cup_2026_rules.json``) as the single source of truth, so these tests pin the
contracts the simulator depends on: the full M73→M104 match DAG, the exact Round-of-16 crossings
(a regression guard against re-introducing naive sequential pairing), the perfect-binary-tree
occupancy geometry, and that Annexe C is 495 same-group-protected bijections.
"""

from __future__ import annotations

from core import knockout


def _spec() -> knockout.KnockoutSpec:
    return knockout.build_spec(knockout.load_rules())


# --------------------------------------------------------------------------------------
# build_spec — the match DAG
# --------------------------------------------------------------------------------------
def test_build_spec_match_counts_and_ids():
    spec = _spec()
    by_round: dict[str, list] = {}
    for m in spec.matches:
        by_round.setdefault(m.round, []).append(m)
    assert len(by_round["R32"]) == 16
    assert len(by_round["R16"]) == 8
    assert len(by_round["QF"]) == 4
    assert len(by_round["SF"]) == 2
    assert len(by_round["3P"]) == 1
    assert len(by_round["Final"]) == 1

    ids = [m.id for m in spec.matches]
    assert ids == [f"M{n}" for n in range(73, 105)]  # M73 … M104, unique and in dependency order
    assert len(set(ids)) == 32
    assert spec.final_id == "M104"


def test_round_of_32_winner_tokens_and_final():
    spec = _spec()
    by_id = {m.id: m for m in spec.matches}
    # Each R32..SF match advances its winner as W{id}; the 3P match and the final advance nobody.
    assert by_id["M73"].win_token == "W73"
    assert by_id["M101"].win_token == "W101"
    assert by_id["M103"].win_token is None  # third-place match
    assert by_id["M104"].win_token is None  # final → champion
    # The semis' losers feed the third-place match.
    assert by_id["M101"].lose_token == "L101"
    assert {by_id["M103"].a, by_id["M103"].b} == {"L101", "L102"}


def test_round_of_16_crossings_match_the_file():
    # Guards against re-introducing naive sequential round pairing: the official R16 crosses the R32
    # winners (M89 = W74·W77, M90 = W73·W75), it does not pair W73·W74.
    spec = _spec()
    by_id = {m.id: m for m in spec.matches}
    assert {by_id["M89"].a, by_id["M89"].b} == {"W74", "W77"}
    assert {by_id["M90"].a, by_id["M90"].b} == {"W73", "W75"}


# --------------------------------------------------------------------------------------
# occupancy_layout — the perfect-binary-tree geometry
# --------------------------------------------------------------------------------------
def test_occupancy_layout_is_a_perfect_binary_tree():
    spec = _spec()
    layout = knockout.occupancy_layout(spec)
    sizes = {"R32": 16, "R16": 8, "QF": 4, "SF": 2, "Final": 1}

    for rk, n in sizes.items():
        rows = layout.rows_by_round[rk]
        assert len(rows) == n
        assert all(rows), f"{rk} has an unfilled row"
        assert len(set(rows)) == n, f"{rk} rows are not a permutation"
        for i, mid in enumerate(rows):
            assert layout.match_row[mid] == (rk, i)

    # r32_slot_order is the 32 participant slots in bracket order: match at row r owns slots 2r/2r+1.
    assert len(layout.r32_slot_order) == 32
    for r, mid in enumerate(layout.rows_by_round["R32"]):
        assert layout.r32_slot_order[2 * r] == (mid, "a")
        assert layout.r32_slot_order[2 * r + 1] == (mid, "b")


def test_occupancy_layout_feeders_are_rows_2m_and_2m_plus_1_every_round():
    # The identity core.bracket relies on: round R's match at row r is fed by the previous round's
    # matches at rows 2r and 2r+1 (in team_a/team_b order). Checked for EVERY round, since the 2026
    # bracket crosses at every round, not only R32→R16.
    spec = _spec()
    layout = knockout.occupancy_layout(spec)
    by_id = {m.id: m for m in spec.matches}
    produces = {m.win_token: m for m in spec.matches if m.win_token}
    for rk, prev in [("R16", "R32"), ("QF", "R16"), ("SF", "QF"), ("Final", "SF")]:
        rows_prev = layout.rows_by_round[prev]
        for mid in layout.rows_by_round[rk]:
            m = by_id[mid]
            r = layout.match_row[mid][1]
            assert rows_prev[2 * r] == produces[m.a].id
            assert rows_prev[2 * r + 1] == produces[m.b].id


# --------------------------------------------------------------------------------------
# Annexe C — the 495-row third-place matrix
# --------------------------------------------------------------------------------------
def test_annex_assignment_known_rows():
    spec = _spec()
    # The eight third-place groups A..H qualify (matrix key "ABCDEFGH").
    assert knockout.annex_assignment(spec.annex_matrix, set("ABCDEFGH")) == {
        "1A": "3H",
        "1B": "3G",
        "1D": "3B",
        "1E": "3C",
        "1G": "3A",
        "1I": "3F",
        "1K": "3D",
        "1L": "3E",
    }
    # The eight third-place groups E..L qualify (matrix key "EFGHIJKL").
    assert knockout.annex_assignment(spec.annex_matrix, set("EFGHIJKL")) == {
        "1A": "3E",
        "1B": "3J",
        "1D": "3I",
        "1E": "3F",
        "1G": "3H",
        "1I": "3G",
        "1K": "3L",
        "1L": "3K",
    }
    # The key is order-independent (sorted internally).
    assert knockout.annex_assignment(spec.annex_matrix, "HGFEDCBA") == knockout.annex_assignment(
        spec.annex_matrix, "ABCDEFGH"
    )


def test_annex_matrix_is_495_same_group_protected_bijections():
    spec = _spec()
    assert len(spec.annex_matrix) == 495
    eligible_winners = {"1A", "1B", "1D", "1E", "1G", "1I", "1K", "1L"}
    for key, assignment in spec.annex_matrix.items():
        assert len(key) == 8
        assert set(assignment) == eligible_winners  # all eight Annexe-eligible winners are routed
        thirds = list(assignment.values())
        assert all(t.startswith("3") for t in thirds)
        assert len(set(thirds)) == 8  # a bijection onto eight distinct third-placed teams
        assert {t[1] for t in thirds} == set(
            key
        )  # opponents come from exactly the qualifying groups
        for winner, third in assignment.items():
            assert winner[1] != third[1]  # same-group protection: 1X never meets 3X
