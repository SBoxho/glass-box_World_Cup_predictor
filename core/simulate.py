"""Monte Carlo simulation of the 48-team / 2026 tournament format.

The format is new for 2026 and easy to get subtly wrong, so the structure is explicit and
unit-tested (see ``tests/test_simulate_format.py``):

* 12 groups of 4 play a round robin (6 matches each, 72 total).
* The top 2 of every group (24 teams) **plus the 8 best third-placed teams** advance to a
  **Round of 32**.
* Single elimination from there: R32 -> R16 -> QF -> SF -> Final. One champion.

Match outcomes are sampled from the model's calibrated probabilities. Scorelines (needed only
for the points -> goal-difference -> goals-scored tiebreakers) are drawn from independent
Poissons whose means are shifted by the model's predicted supremacy (P(home) - P(away)); a
drawn knockout is resolved in favour of the stronger side (a proxy for extra time + penalties).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core import bracket as bracket_mod
from core.config import SEED

# Round-robin pairing order for the 4 teams in a group (by their index 0..3).
GROUP_PAIRINGS = [(0, 1), (2, 3), (0, 2), (1, 3), (0, 3), (1, 2)]

# Scoreline (tiebreaker) model — see module docstring.
GOAL_BASE = 1.35
GOAL_SUPREMACY = 1.1
GOAL_CLIP = (0.2, 5.0)
SCORE_MAX_TRIES = 24

STAGES = ["R32", "R16", "QF", "SF", "Final", "Champion"]


def select_best_thirds(thirds: list[dict], k: int = 8) -> list[dict]:
    """Return the ``k`` best third-placed teams, ranked by points, goal difference, goals for.

    Each input dict must carry ``pts``, ``gd``, ``gf`` and a precomputed ``tiebreak`` random
    value (so ties resolve deterministically within a simulation). This is the piece the 2026
    format hinges on, so it is a standalone, directly tested function.
    """
    ranked = sorted(thirds, key=lambda t: (t["pts"], t["gd"], t["gf"], t["tiebreak"]), reverse=True)
    return ranked[:k]


@dataclass
class SimulationResult:
    """Aggregated outcome of an N-run simulation."""

    table: pd.DataFrame  # one row per team: group + probability of reaching each stage
    most_likely_final: tuple[str, str]
    most_likely_final_prob: float
    n_sims: int
    # Per-slot bracket projection (see :mod:`core.bracket`): for every knockout position, which
    # teams are likely to fill it and who advances. ``None`` only if bracket tracking was disabled.
    bracket: dict | None = None


class TournamentSimulator:
    """Precompute every matchup probability once, then sample the bracket ``n_sims`` times.

    ``predictor`` only needs a ``predict(home, away, neutral, is_host_home) -> {H,D,A}`` method,
    so the simulator is decoupled from the model internals (and easy to stub in tests).
    """

    def __init__(self, predictor, wc2026: dict, seed: int = SEED):
        self.predictor = predictor
        self.wc = wc2026
        self.rng = np.random.default_rng(seed)
        self.groups: dict[str, list[str]] = wc2026["groups"]
        self.teams: list[str] = [t for teams in self.groups.values() for t in teams]
        self.idx = {t: i for i, t in enumerate(self.teams)}
        self.group_of = {t: g for g, teams in self.groups.items() for t in teams}
        self.host_groups = wc2026.get("host_groups", {})
        self.r32_template = wc2026["r32_template"]
        self.best_thirds_count = wc2026.get("best_thirds_count", 8)
        self._known = self._index_known_results(wc2026.get("known_results", []))
        self._prepare()

    # -- precompute -----------------------------------------------------------------
    def _index_known_results(self, known: list[dict]) -> dict:
        out = {}
        for k in known:
            out[frozenset((k["home"], k["away"]))] = k
        return out

    def _group_fixtures(self):
        """For each group, the 6 oriented fixtures with their outcome probabilities + goal means."""
        fixtures = {}
        for g, teams in self.groups.items():
            host = next((t for t in teams if self.host_groups.get(t) == g), None)
            flist = []
            for i, j in GROUP_PAIRINGS:
                a, b = teams[i], teams[j]
                if b == host:  # ensure the host is the home side of its matches
                    a, b = b, a
                neutral = a != host
                is_host = int(a == host)
                probs = self.predictor.predict(a, b, neutral=neutral, is_host_home=is_host)
                p = (probs["H"], probs["D"], probs["A"])
                lam_a, lam_b = self._goal_means(*p)
                flist.append((a, b, p, lam_a, lam_b))
            fixtures[g] = flist
        return fixtures

    def _goal_means(self, pH: float, pD: float, pA: float) -> tuple[float, float]:
        sup = pH - pA
        lam_a = float(np.clip(GOAL_BASE + GOAL_SUPREMACY * sup, *GOAL_CLIP))
        lam_b = float(np.clip(GOAL_BASE - GOAL_SUPREMACY * sup, *GOAL_CLIP))
        return lam_a, lam_b

    def _prepare(self):
        self.fixtures = self._group_fixtures()
        # Pairwise probability that row-team beats col-team in a neutral knockout tie.
        n = len(self.teams)
        adv = np.full((n, n), 0.5)
        pairs = [(a, b) for a in range(n) for b in range(a + 1, n)]
        team_pairs = [(self.teams[a], self.teams[b]) for a, b in pairs]
        if hasattr(self.predictor, "predict_pairs_neutral"):
            probs_list = self.predictor.predict_pairs_neutral(team_pairs)
        else:
            probs_list = [
                self.predictor.predict(a, b, neutral=True, is_host_home=0) for a, b in team_pairs
            ]
        for (a, b), probs in zip(pairs, probs_list, strict=True):
            pH, pD, pA = probs["H"], probs["D"], probs["A"]
            denom = pH + pA
            share = pH / denom if denom > 0 else 0.5
            p_a = pH + pD * share  # a wins, or draws then prevails in ET/penalties
            adv[a, b] = p_a
            adv[b, a] = 1.0 - p_a
        self.adv = adv

    # -- one simulation -------------------------------------------------------------
    def _sample_score(self, p, lam_a, lam_b) -> tuple[int, int]:
        """Sample (goals_a, goals_b) consistent with a sampled {H,D,A} outcome."""
        outcome = self.rng.choice(3, p=p)  # 0=a wins, 1=draw, 2=b wins
        for _ in range(SCORE_MAX_TRIES):
            ga, gb = int(self.rng.poisson(lam_a)), int(self.rng.poisson(lam_b))
            s = 0 if ga > gb else (1 if ga == gb else 2)
            if s == outcome:
                return ga, gb
        return {0: (1, 0), 1: (1, 1), 2: (0, 1)}[int(outcome)]

    def _play_group(self, g: str):
        """Simulate one group; return ordered standings as a list of dicts (1st..4th)."""
        teams = self.groups[g]
        stat = {t: {"pts": 0, "gd": 0, "gf": 0} for t in teams}
        for a, b, p, lam_a, lam_b in self.fixtures[g]:
            known = self._known.get(frozenset((a, b)))
            if known is not None:
                ga, gb = known["home_score"], known["away_score"]
                if known["home"] != a:  # stored orientation is reversed
                    ga, gb = gb, ga
            else:
                ga, gb = self._sample_score(p, lam_a, lam_b)
            stat[a]["gf"] += ga
            stat[b]["gf"] += gb
            stat[a]["gd"] += ga - gb
            stat[b]["gd"] += gb - ga
            if ga > gb:
                stat[a]["pts"] += 3
            elif ga == gb:
                stat[a]["pts"] += 1
                stat[b]["pts"] += 1
            else:
                stat[b]["pts"] += 3
        standings = [
            {"team": t, "group": g, "tiebreak": self.rng.random(), **stat[t]} for t in teams
        ]
        standings.sort(key=lambda s: (s["pts"], s["gd"], s["gf"], s["tiebreak"]), reverse=True)
        return standings

    def _assign_thirds(self, qualifying_thirds: list[dict]) -> dict[str, str]:
        """Map third-place qualifiers to the template's T_n slots, avoiding a team meeting its
        own group's winner in the Round of 32 (FIFA's exact slotting table is approximated)."""
        # Winner-group each T_n slot faces, read off the committed template.
        faces = {}
        for a, b in self.r32_template:
            for tok, other in ((a, b), (b, a)):
                if tok.startswith("T_"):
                    n = int(tok.split("_")[1])
                    faces[n] = other.split("_")[1] if other.startswith("W_") else None
        assignment = {}
        remaining = list(qualifying_thirds)
        for n in range(1, self.best_thirds_count + 1):
            avoid = faces.get(n)
            pick = next((t for t in remaining if t["group"] != avoid), remaining[0])
            assignment[f"T_{n}"] = pick["team"]
            remaining.remove(pick)
        return assignment

    def _resolve_slot(self, token: str, winners, runners, thirds_slot) -> str:
        kind, g = token.split("_")
        if kind == "W":
            return winners[g]
        if kind == "R":
            return runners[g]
        return thirds_slot[token]

    def _knockout(self, a: str, b: str) -> str:
        p = self.adv[self.idx[a], self.idx[b]]
        return a if self.rng.random() < p else b

    def simulate_once(self) -> dict:
        """Play one full tournament; return the set of teams reaching each stage + the final."""
        winners, runners, thirds = {}, {}, []
        for g in self.groups:
            standings = self._play_group(g)
            winners[g] = standings[0]["team"]
            runners[g] = standings[1]["team"]
            thirds.append(standings[2])

        best_thirds = select_best_thirds(thirds, self.best_thirds_count)
        thirds_slot = self._assign_thirds(best_thirds)

        reached = {s: [] for s in STAGES}
        r32 = [t["team"] for t in best_thirds] + list(winners.values()) + list(runners.values())
        reached["R32"] = r32

        # Build R32 matchups from the template, then play down the bracket.
        round_pairs = [
            (
                self._resolve_slot(x, winners, runners, thirds_slot),
                self._resolve_slot(y, winners, runners, thirds_slot),
            )
            for x, y in self.r32_template
        ]
        self._last_r32_pairs = list(round_pairs)
        for stage in ["R16", "QF", "SF", "Final", "Champion"]:
            if stage == "Champion":  # round_pairs is now the single final matchup
                self._last_final = tuple(sorted(round_pairs[0]))
            won = [self._knockout(a, b) for a, b in round_pairs]
            reached[stage] = won
            if stage != "Champion":
                round_pairs = [(won[i], won[i + 1]) for i in range(0, len(won), 2)]
        return reached

    # -- aggregate ------------------------------------------------------------------
    def run(self, n_sims: int = 20000, seed: int | None = None) -> SimulationResult:
        """Run ``n_sims`` tournaments and aggregate per-team stage probabilities.

        Pass ``seed`` to reset the RNG first, so the same call is reproducible (the app does
        this on each run); leave it ``None`` to continue the existing random stream.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        n = len(self.teams)
        counts = {s: np.zeros(n) for s in STAGES}
        final_pairs: dict[tuple, int] = {}

        # Per-position occupancy: occ[round][slot, team] counts how many tournaments put each team in
        # that bracket slot. ``reached[stage]`` is already in bracket order; the Round-of-32 *slot*
        # order comes from the resolved template pairs (``_last_r32_pairs``). See :mod:`core.bracket`.
        occ = {k: np.zeros((bracket_mod.PARTICIPANTS[k], n)) for k in bracket_mod.PARTICIPANTS}
        ko_rows = {k: np.arange(bracket_mod.PARTICIPANTS[k]) for k in ["R16", "QF", "SF", "Final"]}
        r32_rows = np.arange(32)

        for _ in range(n_sims):
            reached = self.simulate_once()
            for s in STAGES:
                for t in reached[s]:
                    counts[s][self.idx[t]] += 1
            flat32 = [t for pair in self._last_r32_pairs for t in pair]
            occ["R32"][r32_rows, [self.idx[t] for t in flat32]] += 1
            for s in ["R16", "QF", "SF", "Final"]:
                occ[s][ko_rows[s], [self.idx[t] for t in reached[s]]] += 1
            occ["Champion"][0, self.idx[reached["Champion"][0]]] += 1
            fp = self._last_final
            final_pairs[fp] = final_pairs.get(fp, 0) + 1

        table = pd.DataFrame({"team": self.teams, "group": [self.group_of[t] for t in self.teams]})
        for s in STAGES:
            table[s] = counts[s] / n_sims
        table = table.sort_values("Champion", ascending=False).reset_index(drop=True)

        best_final = max(final_pairs.items(), key=lambda kv: kv[1])
        r32_tokens = [tok for pair in self.r32_template for tok in pair]
        return SimulationResult(
            table=table,
            most_likely_final=best_final[0],
            most_likely_final_prob=best_final[1] / n_sims,
            n_sims=n_sims,
            bracket=bracket_mod.build_bracket(occ, self.teams, n_sims, r32_tokens),
        )
