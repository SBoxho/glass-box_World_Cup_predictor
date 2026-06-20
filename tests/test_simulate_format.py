"""Guardrail 3 — 48-team / Round-of-32 format correctness.

The 2026 format (48 teams, 12 groups, top-2 + 8 best thirds -> R32) is new and the highest-risk
part of the project, so its structure is asserted explicitly: exactly 48 teams enter, exactly 32
reach the R32 (24 group qualifiers + 8 best thirds), the best-third ranking is correct, no team
meets its own group in the R32, and the bracket yields exactly one champion.
"""

from __future__ import annotations

import math

import numpy as np

from core import simulate
from core.ingest import load_wc2026


class StubPredictor:
    """Deterministic stand-in: outcome probabilities from per-team latent strengths.

    The format test is about structure, not model realism — so a cheap strength-based predictor
    keeps it fast and independent of any trained artifact or network access.
    """

    def __init__(self, teams, seed: int = 1):
        rng = np.random.default_rng(seed)
        self.strength = {t: float(rng.normal(0, 1)) for t in teams}

    def predict(self, home, away, neutral=True, is_host_home=0):
        s = self.strength[home] - self.strength[away] + (0.0 if neutral else 0.3)
        eH, eA, eD = math.exp(s), math.exp(-s), math.exp(0.0) * 1.3
        z = eH + eA + eD
        return {"H": eH / z, "D": eD / z, "A": eA / z}


def _make_sim(seed: int = 0):
    wc = load_wc2026()
    teams = [t for grp in wc["groups"].values() for t in grp]
    return simulate.TournamentSimulator(StubPredictor(teams), wc, seed=seed), wc


def test_exactly_48_teams_enter():
    sim, _ = _make_sim()
    assert len(sim.teams) == 48
    assert len(set(sim.teams)) == 48


def test_single_simulation_bracket_shape():
    sim, wc = _make_sim(seed=3)
    reached = sim.simulate_once()

    # 24 group qualifiers + 8 best thirds = 32, all distinct.
    assert len(reached["R32"]) == 32
    assert len(set(reached["R32"])) == 32

    # Single-elimination halving down to one champion.
    assert [len(reached[s]) for s in ["R16", "QF", "SF", "Final", "Champion"]] == [16, 8, 4, 2, 1]

    # Exactly two qualifiers per group come from the top-2; the remaining 8 are thirds.
    groups_of = sim.group_of
    qualifiers = reached["R32"]
    per_group = dict.fromkeys(wc["groups"], 0)
    for t in qualifiers:
        per_group[groups_of[t]] += 1
    # Eight groups send 3 (winner + runner + a third); four groups send 2.
    assert sorted(per_group.values()) == [2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3]
    assert sum(per_group.values()) == 32


def test_no_same_group_matchups_in_r32():
    # Now structurally guaranteed by the official bracket + Annexe C (no team meets its own group in
    # the R32), so assert it holds across many seeds and repeated draws, not just one.
    for seed in range(6):
        sim, _ = _make_sim(seed=seed)
        for _ in range(20):
            reached = sim.simulate_once()
            assert len(set(reached["R32"])) == 32  # every slate is 32 distinct teams
            for a, b in sim._last_r32_pairs:
                assert sim.group_of[a] != sim.group_of[b], f"same-group R32 rematch: {a} vs {b}"


def test_official_group_tiebreaker_prefers_head_to_head():
    # Three teams level on points. Overall goal difference says A > B > C, but the head-to-head
    # mini-table (C beat both A and B; A beat B) says C > A > B — the official order follows H2H.
    standings = [
        {"team": "A", "group": "Z", "pts": 6, "gd": 10, "gf": 12, "tiebreak": 0.5},
        {"team": "B", "group": "Z", "pts": 6, "gd": 5, "gf": 8, "tiebreak": 0.5},
        {"team": "C", "group": "Z", "pts": 6, "gd": 1, "gf": 3, "tiebreak": 0.5},
        {"team": "D", "group": "Z", "pts": 0, "gd": -16, "gf": 0, "tiebreak": 0.5},
    ]
    results = [
        ("C", "A", 1, 0),  # head-to-head among the tied set
        ("C", "B", 1, 0),
        ("A", "B", 1, 0),
        ("A", "D", 5, 0),  # blow-out wins over D explain the misleading overall GD
        ("B", "D", 4, 0),
        ("C", "D", 1, 0),
    ]
    order = [s["team"] for s in simulate.order_standings(standings, results)]
    assert order == ["C", "A", "B", "D"]


def test_fifa_ranking_breaks_an_otherwise_exact_tie():
    # Two teams identical on every on-pitch criterion (drawn head-to-head, equal overall GD/GF) →
    # the better FIFA position decides, ahead of the random tiebreak.
    standings = [
        {"team": "P", "group": "Z", "pts": 3, "gd": 0, "gf": 1, "tiebreak": 0.1},
        {"team": "Q", "group": "Z", "pts": 3, "gd": 0, "gf": 1, "tiebreak": 0.9},
    ]
    results = [("P", "Q", 1, 1)]  # drawn head-to-head — no separation before FIFA rank
    order = [
        s["team"] for s in simulate.order_standings(standings, results, fifa_rank={"P": 5, "Q": 20})
    ]
    assert order == ["P", "Q"]  # P (5th) beats Q (20th), despite P's lower random tiebreak value
    # Without the FIFA criterion the random tiebreak decides instead (Q's 0.9 > P's 0.1).
    assert [s["team"] for s in simulate.order_standings(standings, results)] == ["Q", "P"]


def test_select_best_thirds_ranking():
    # 12 third-placed teams with strictly decreasing quality A..L.
    thirds = [
        {
            "team": chr(65 + i),
            "group": chr(65 + i),
            "pts": 12 - i,
            "gd": 6 - i,
            "gf": 8 - i,
            "tiebreak": 0.5,
        }
        for i in range(12)
    ]
    best = simulate.select_best_thirds(thirds, 8)
    assert len(best) == 8
    assert [t["team"] for t in best] == list("ABCDEFGH")  # the 8 strongest, in order

    # Ties on pts/gd/gf fall back to the deterministic tiebreak value.
    tied = [
        {"team": "X", "group": "X", "pts": 4, "gd": 0, "gf": 2, "tiebreak": 0.9},
        {"team": "Y", "group": "Y", "pts": 4, "gd": 0, "gf": 2, "tiebreak": 0.1},
    ]
    assert simulate.select_best_thirds(tied, 1)[0]["team"] == "X"


def test_aggregate_probabilities_are_consistent():
    sim, _ = _make_sim(seed=7)
    res = sim.run(n_sims=150)
    # Exactly one champion per simulation -> Champion column sums to 1.0.
    assert math.isclose(res.table["Champion"].sum(), 1.0, abs_tol=1e-9)
    # Exactly 32 qualifiers per simulation -> R32 column sums to 32.
    assert math.isclose(res.table["R32"].sum(), 32.0, abs_tol=1e-9)
    # Stage probabilities are monotonically non-increasing per team (can't reach SF without QF).
    for _, row in res.table.iterrows():
        chain = [row[s] for s in ["R32", "R16", "QF", "SF", "Final", "Champion"]]
        assert all(chain[i] >= chain[i + 1] - 1e-9 for i in range(len(chain) - 1))
    assert len(res.most_likely_final) == 2
