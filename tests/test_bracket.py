"""Guardrail — the bracket projection and its editorial highlights are correct.

The bracket view is only as honest as the per-slot occupancy it is drawn from, so the geometry that
links a match to its participants and its advancer is pinned with hand-built counts (deterministic,
no model), and the highlight derivations (Cinderella, closest call, traced path) are checked on small
synthetic inputs. A single end-to-end run then confirms the simulator wires real counts through with
the right shape and stays consistent with its own aggregate table.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from core import bracket as cb
from core import simulate
from core.ingest import load_wc2026
from tests.test_simulate_format import StubPredictor


# --------------------------------------------------------------------------------------
# token_label — Round-of-32 slot identities
# --------------------------------------------------------------------------------------
def test_token_label():
    # Legacy template tokens (kept for backward compatibility).
    assert cb.token_label("W_A") == "Winners Group A"
    assert cb.token_label("R_K") == "Runners-up Group K"
    assert cb.token_label("T_1") == "3rd place #1"
    assert cb.token_label(None) is None
    assert cb.token_label("garbage") is None


def test_token_label_official_tokens():
    # Official rules-file tokens (core.knockout): 1X / 2X / 3X and the Annexe C best-third slot.
    assert cb.token_label("1E") == "Winners Group E"
    assert cb.token_label("2A") == "Runners-up Group A"
    assert cb.token_label("3C") == "3rd place Group C"
    assert cb.token_label("ANNEX_C_FOR_1E") == "Best 3rd (vs Group E)"
    assert cb.token_label("W74") is None  # not a Round-of-32 slot identity


# --------------------------------------------------------------------------------------
# build_bracket — slot occupancy + the occupancy-is-advancement-one-round-on identity
# --------------------------------------------------------------------------------------
def _empty_occ(n_teams: int) -> dict:
    return {k: np.zeros((cb.PARTICIPANTS[k], n_teams)) for k in cb.PARTICIPANTS}


def test_build_bracket_maps_slots_and_advancement():
    teams = [f"T{i}" for i in range(40)]
    n_sims = 100
    occ = _empty_occ(len(teams))
    # R32 match 0: top slot split 60/40 between T0 and T1; bottom slot all T2.
    occ["R32"][0, 0] = 60
    occ["R32"][0, 1] = 40
    occ["R32"][1, 2] = 100
    # The winner of R32 match 0 is participant slot 0 of R16 → a 50/50 between T0 and T2.
    occ["R16"][0, 0] = 50
    occ["R16"][0, 2] = 50

    b = cb.build_bracket(occ, teams, n_sims, r32_tokens=["W_A", "T_1"] + [None] * 30)

    assert [len(r["matches"]) for r in b["rounds"]] == [16, 8, 4, 2, 1]
    m0 = b["rounds"][0]["matches"][0]
    assert m0["top"] == [{"team": "T0", "p": 0.6}, {"team": "T1", "p": 0.4}]
    assert m0["bottom"] == [{"team": "T2", "p": 1.0}]
    # advance reads the NEXT round's slot m — occupancy and advancement are the same array, shifted.
    assert {(d["team"], d["p"]) for d in m0["advance"]} == {("T0", 0.5), ("T2", 0.5)}
    # Round-of-32 slots carry their template identity.
    assert m0["top_label"] == "Winners Group A"
    assert m0["bottom_label"] == "3rd place #1"


def test_build_bracket_truncates_and_thresholds():
    teams = [f"T{i}" for i in range(40)]
    occ = _empty_occ(len(teams))
    # A noisy slot: one dominant team, three faint ones (one below eps).
    occ["R32"][0, 0] = 970
    occ["R32"][0, 1] = 20
    occ["R32"][0, 2] = 8
    occ["R32"][0, 3] = 2  # 0.2% — below the default 0.4% floor
    b = cb.build_bracket(occ, teams, 1000, top_k=2)
    top = b["rounds"][0]["matches"][0]["top"]
    assert [d["team"] for d in top] == ["T0", "T1"]  # top_k=2 keeps only the two largest
    # And with a higher top_k the sub-eps entry is still dropped.
    top_all = cb.build_bracket(occ, teams, 1000, top_k=10)["rounds"][0]["matches"][0]["top"]
    assert [d["team"] for d in top_all] == ["T0", "T1", "T2"]  # T3 (0.2%) filtered out


def test_champion_node_reads_the_champion_slot():
    teams = ["A", "B", "C"]
    occ = _empty_occ(3)
    occ["Champion"][0, 0] = 70
    occ["Champion"][0, 1] = 30
    b = cb.build_bracket(occ, teams, 100)
    assert b["champion"] == [{"team": "A", "p": 0.7}, {"team": "B", "p": 0.3}]


# --------------------------------------------------------------------------------------
# Highlights — closest call / Cinderella / traced path
# --------------------------------------------------------------------------------------
def _mini_bracket() -> dict:
    return {
        "rounds": [
            {
                "key": "SF",
                "title": "Semi-finals",
                "matches": [
                    {  # a near coin-flip between two near-certain participants
                        "round": "SF",
                        "index": 0,
                        "top": [{"team": "A", "p": 0.9}],
                        "bottom": [{"team": "B", "p": 0.9}],
                        "advance": [{"team": "A", "p": 0.46}, {"team": "B", "p": 0.44}],
                    },
                    {  # a lopsided match — should not win "most uncertain"
                        "round": "SF",
                        "index": 1,
                        "top": [{"team": "C", "p": 0.9}],
                        "bottom": [{"team": "D", "p": 0.9}],
                        "advance": [{"team": "C", "p": 0.82}, {"team": "D", "p": 0.10}],
                    },
                ],
            }
        ]
    }


def test_most_uncertain_match_picks_the_coin_flip():
    res = cb.most_uncertain_match(_mini_bracket())
    assert res["index"] == 0
    assert {res["team_a"], res["team_b"]} == {"A", "B"}
    assert math.isclose(res["p_a"], 0.46 / 0.90)
    assert math.isclose(res["p_b"], 0.44 / 0.90)


def test_most_uncertain_match_skips_undetermined_pairings():
    # Both slots are too speculative (below min_determined) → no qualifying match.
    bracket = {
        "rounds": [
            {
                "key": "QF",
                "title": "Quarter-finals",
                "matches": [
                    {
                        "round": "QF",
                        "index": 0,
                        "top": [{"team": "A", "p": 0.2}],
                        "bottom": [{"team": "B", "p": 0.2}],
                        "advance": [{"team": "A", "p": 0.1}, {"team": "B", "p": 0.1}],
                    }
                ],
            }
        ]
    }
    assert cb.most_uncertain_match(bracket) is None


def test_biggest_upset_finds_the_strongest_non_seed_deep_run():
    table = pd.DataFrame(
        {
            "team": ["Strong", "Mid", "Dark"],
            "R32": [1.0, 1.0, 1.0],
            "R16": [0.9, 0.5, 0.4],
            "QF": [0.7, 0.3, 0.25],
            "SF": [0.5, 0.12, 0.08],
            "Final": [0.3, 0.04, 0.02],
            "Champion": [0.2, 0.01, 0.0],
        }
    )
    ratings = {"Strong": 2000.0, "Mid": 1800.0, "Dark": 1700.0}
    # seed_cutoff=1 → only "Strong" is a seed; the underdog with the best semis odds is "Mid".
    res = cb.biggest_upset(table, ratings, seed_cutoff=1)
    assert res["team"] == "Mid"
    assert res["stage"] == "SF"
    assert math.isclose(res["p"], 0.12)
    assert res["elo_rank"] == 2


def test_biggest_upset_returns_none_without_a_real_run():
    table = pd.DataFrame(
        {
            "team": ["Strong", "Weak"],
            "R32": [1.0, 1.0],
            "R16": [0.9, 0.01],
            "QF": [0.7, 0.0],
            "SF": [0.5, 0.0],
            "Final": [0.3, 0.0],
            "Champion": [0.2, 0.0],
        }
    )
    ratings = {"Strong": 2000.0, "Weak": 1500.0}
    assert cb.biggest_upset(table, ratings, seed_cutoff=1) is None  # no underdog clears the floor


def test_team_path_traces_the_strongest_slot_per_round():
    bracket = {
        "rounds": [
            {
                "key": "R32",
                "title": "Round of 32",
                "matches": [
                    {"round": "R32", "index": 0, "top": [], "bottom": [], "advance": []},
                    {
                        "round": "R32",
                        "index": 1,
                        "top": [{"team": "X", "p": 0.55}],
                        "bottom": [{"team": "Y", "p": 0.7}],
                        "advance": [{"team": "Y", "p": 0.5}],
                    },
                ],
            },
            {
                "key": "R16",
                "title": "Round of 16",
                "matches": [
                    {
                        "round": "R16",
                        "index": 0,
                        "top": [{"team": "X", "p": 0.03}],  # below min_p → not traced
                        "bottom": [{"team": "Z", "p": 0.9}],
                        "advance": [{"team": "Z", "p": 0.8}],
                    }
                ],
            },
        ]
    }
    path = cb.team_path(bracket, "X")
    assert path["R32"] == {"index": 1, "side": "top", "p": 0.55}
    assert "R16" not in path  # X's only R16 appearance (0.03) is below the noise floor
    assert cb.team_path(bracket, None) == {}


# --------------------------------------------------------------------------------------
# End-to-end — the simulator wires real counts through with the right shape + consistency
# --------------------------------------------------------------------------------------
def test_run_attaches_consistent_bracket():
    wc = load_wc2026()
    teams = [t for grp in wc["groups"].values() for t in grp]
    sim = simulate.TournamentSimulator(StubPredictor(teams), wc, seed=0)
    res = sim.run(n_sims=300, seed=1)

    b = res.bracket
    assert b is not None
    assert [r["key"] for r in b["rounds"]] == ["R32", "R16", "QF", "SF", "Final"]
    assert [len(r["matches"]) for r in b["rounds"]] == [16, 8, 4, 2, 1]

    # The bracket's projected champion is the same team (and probability) as the aggregate table's.
    top_row = res.table.iloc[0]
    assert b["champion"][0]["team"] == top_row["team"]
    assert math.isclose(b["champion"][0]["p"], float(top_row["Champion"]), abs_tol=1e-9)

    # Element 0 of each slot is the coherent lead (which may not be the global argmax), so the list
    # as a whole need not be descending — but the tail is, and each slot stays a probability over
    # teams (sums to ≤ 1). Probabilities themselves are never altered by the coherent projection.
    for rnd in b["rounds"]:
        for match in rnd["matches"]:
            for side in ("top", "bottom", "advance"):
                ps = [d["p"] for d in match[side]]
                assert ps[1:] == sorted(ps[1:], reverse=True)
                assert sum(ps) <= 1.0 + 1e-9


def test_projection_has_no_duplicate_team_within_a_round():
    # Issue-1 regression: a dominant team must not lead two slots of the same round (e.g. appear as
    # both a group's winner and its runner-up in the Round of 32). The coherent projection forbids it.
    wc = load_wc2026()
    teams = [t for grp in wc["groups"].values() for t in grp]
    sim = simulate.TournamentSimulator(StubPredictor(teams), wc, seed=0)
    res = sim.run(n_sims=500, seed=2)
    for rnd in res.bracket["rounds"]:
        leaders = [
            match[side][0]["team"]
            for match in rnd["matches"]
            for side in ("top", "bottom")
            if match[side]
        ]
        assert len(leaders) == len(set(leaders)), f"duplicate slot leader in round {rnd['key']}"


def test_advance_identity_holds_under_official_geometry():
    # advance for match m reads the NEXT round's participant slot m — independent of which physical
    # matches permute onto which rows. Hand-built occupancy with a known winner per Round-of-32 slot.
    teams = [f"T{i}" for i in range(40)]
    n = 1000
    occ = _empty_occ(len(teams))
    for m in range(16):
        occ["R32"][2 * m, 2 * m] = n  # match m: top slot all T(2m)
        occ["R32"][2 * m + 1, 2 * m + 1] = n  # bottom slot all T(2m+1)
        occ["R16"][m, 2 * m] = n  # winner advancing into R16 slot m is the top team T(2m)
    b = cb.build_bracket(occ, teams, n)
    for m in range(16):
        adv = b["rounds"][0]["matches"][m]["advance"]
        assert adv == [{"team": f"T{2 * m}", "p": 1.0}]
