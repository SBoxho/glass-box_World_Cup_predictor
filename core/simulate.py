"""Monte Carlo simulation of the 48-team / 2026 tournament format.

The format is new for 2026 and easy to get subtly wrong, so the structure is explicit and
unit-tested (see ``tests/test_simulate_format.py``):

* 12 groups of 4 play a round robin (6 matches each, 72 total).
* The top 2 of every group (24 teams) **plus the 8 best third-placed teams** advance to a
  **Round of 32**.
* Single elimination from there: R32 -> R16 -> QF -> SF -> Final. One champion.

The knockout phase follows the **official 2026 progression rules**, loaded from
``data/fifa_world_cup_2026_rules.json`` via :mod:`core.knockout` (the single source of truth): the
fixed match DAG (M73→M104) and FIFA's 495-row **Annexe C** matrix, which assigns each Annexe-eligible
group winner its best-third opponent from *which eight groups* supplied the qualifiers. Group
ordering uses the official tiebreakers (head-to-head → overall → FIFA ranking; see
:func:`order_standings`).

Match outcomes are sampled from the model's calibrated probabilities. Scorelines (needed for the
points -> goal-difference -> goals-scored tiebreakers, and the head-to-head mini-tables) are drawn
from independent Poissons whose means are shifted by the model's predicted supremacy
(P(home) - P(away)); a drawn knockout is resolved in favour of the stronger side (a proxy for extra
time + penalties).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core import bracket as bracket_mod
from core import knockout
from core.config import SEED

# Round-robin pairing order for the 4 teams in a group (by their index 0..3).
GROUP_PAIRINGS = [(0, 1), (2, 3), (0, 2), (1, 3), (0, 3), (1, 2)]

# Scoreline (tiebreaker) model — see module docstring.
GOAL_BASE = 1.35
GOAL_SUPREMACY = 1.1
GOAL_CLIP = (0.2, 5.0)
SCORE_MAX_TRIES = 24

STAGES = ["R32", "R16", "QF", "SF", "Final", "Champion"]


def _fifa_rank_key(team: str, fifa_rank: dict[str, int] | None) -> float:
    """Sort key for the FIFA-ranking tiebreaker: a *better* (lower) position sorts higher.

    Returned for ``reverse=True`` sorts, so a smaller rank number yields a larger key; teams with no
    ranking on file get ``-inf`` so they fall behind any ranked team but never crash the sort. When
    ``fifa_rank`` is ``None`` (a bare simulator) every team gets ``-inf`` and the criterion is inert.
    """
    rank = fifa_rank.get(team) if fifa_rank else None
    return -float(rank) if rank is not None else float("-inf")


def select_best_thirds(
    thirds: list[dict], k: int = 8, fifa_rank: dict[str, int] | None = None
) -> list[dict]:
    """Return the ``k`` best third-placed teams, by the official cross-group ranking criteria.

    Order (FIFA Annexe / Article 13): most points → best goal difference → most goals scored →
    higher FIFA-ranking position → a deterministic random ``tiebreak`` (the team-conduct-score and
    "successively earlier rankings" steps are not modelled — see :func:`order_standings`). This is a
    cross-group comparison, so there is no head-to-head step. ``fifa_rank`` maps team → 1-based FIFA
    position (``None`` skips that criterion, keeping bare-simulator tests deterministic). The piece
    the 2026 format hinges on, so it is a standalone, directly tested function.
    """
    ranked = sorted(
        thirds,
        key=lambda t: (
            t["pts"],
            t["gd"],
            t["gf"],
            _fifa_rank_key(t["team"], fifa_rank),
            t["tiebreak"],
        ),
        reverse=True,
    )
    return ranked[:k]


def _head_to_head(block: list[dict], results: list[tuple]) -> dict[str, dict]:
    """Mini-table (pts / gd / gf) over only the matches played *among* a set of tied teams."""
    tied = {s["team"] for s in block}
    h2h = {t: {"pts": 0, "gd": 0, "gf": 0} for t in tied}
    for a, b, ga, gb in results:
        if a in tied and b in tied:
            h2h[a]["gf"] += ga
            h2h[b]["gf"] += gb
            h2h[a]["gd"] += ga - gb
            h2h[b]["gd"] += gb - ga
            if ga > gb:
                h2h[a]["pts"] += 3
            elif ga == gb:
                h2h[a]["pts"] += 1
                h2h[b]["pts"] += 1
            else:
                h2h[b]["pts"] += 3
    return h2h


def order_standings(
    standings: list[dict], results: list[tuple], fifa_rank: dict[str, int] | None = None
) -> list[dict]:
    """Order a group's four teams (1st → 4th) by the official 2026 tiebreakers.

    ``standings`` carries each team's overall ``pts``/``gd``/``gf`` and a random ``tiebreak``;
    ``results`` is the list of ``(home, away, home_goals, away_goals)`` for the six group matches
    (used to build the head-to-head mini-tables). Teams are first grouped by overall points; within
    any points-tied block the order is, in sequence: head-to-head points → h2h goal difference → h2h
    goals → overall goal difference → overall goals → higher FIFA-ranking position → the random
    tiebreak.

    Honest simplifications, documented in the README / *Under the Hood*: the **team-conduct score**
    (no cards are simulated) and the **"successively earlier FIFA rankings"** step are not modelled —
    the random ``tiebreak`` stands in. The "reapply head-to-head to the still-tied subset" rule is
    approximated by a single composite sort over the tied block rather than exact iterative
    re-grouping (a rare Monte-Carlo edge case).
    """
    by_pts: dict[int, list[dict]] = {}
    for s in standings:
        by_pts.setdefault(s["pts"], []).append(s)

    ordered: list[dict] = []
    for pts in sorted(by_pts, reverse=True):
        block = by_pts[pts]
        if len(block) > 1:
            h2h = _head_to_head(block, results)
            block = sorted(
                block,
                key=lambda s: (
                    h2h[s["team"]]["pts"],
                    h2h[s["team"]]["gd"],
                    h2h[s["team"]]["gf"],
                    s["gd"],
                    s["gf"],
                    _fifa_rank_key(s["team"], fifa_rank),
                    s["tiebreak"],
                ),
                reverse=True,
            )
        ordered.extend(block)
    return ordered


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

    def __init__(
        self,
        predictor,
        wc2026: dict,
        seed: int = SEED,
        fifa_rank: dict[str, int] | None = None,
    ):
        self.predictor = predictor
        self.wc = wc2026
        self.rng = np.random.default_rng(seed)
        self.groups: dict[str, list[str]] = wc2026["groups"]
        self.teams: list[str] = [t for teams in self.groups.values() for t in teams]
        self.idx = {t: i for i, t in enumerate(self.teams)}
        self.group_of = {t: g for g, teams in self.groups.items() for t in teams}
        self.host_groups = wc2026.get("host_groups", {})
        self.best_thirds_count = wc2026.get("best_thirds_count", 8)
        # Team -> 1-based FIFA ranking position, the official group/best-third tiebreaker. ``None``
        # (a bare simulator) skips that criterion, keeping the structure tests deterministic.
        self.fifa_rank = fifa_rank
        self._known = self._index_known_results(wc2026.get("known_results", []))
        # Completed knockout ties: a winner per unordered pair, plus the set of teams that have been
        # knocked out. The pair map honors the real result (including upsets) whenever that exact tie
        # recurs in a sim; the eliminated set is the safety net that keeps a team that has actually
        # been knocked out from ever advancing, even if the re-drawn bracket pairs it differently
        # (group tiebreakers are a documented approximation of FIFA's).
        self._known_ko, self._eliminated = self._index_known_ko(wc2026.get("known_ko_results", []))
        self._prepare()

    # -- precompute -----------------------------------------------------------------
    def _index_known_results(self, known: list[dict]) -> dict:
        out = {}
        for k in known:
            out[frozenset((k["home"], k["away"]))] = k
        return out

    def _index_known_ko(self, known: list[dict]) -> tuple[dict, set]:
        """Return ``({frozenset(pair): winner}, {eliminated teams})`` from locked knockout results."""
        by_pair: dict[frozenset, str] = {}
        eliminated: set[str] = set()
        for k in known:
            home, away, winner = k["home"], k["away"], k.get("winner")
            if winner not in (home, away):
                continue  # malformed entry — no usable winner for this tie
            by_pair[frozenset((home, away))] = winner
            eliminated.add(away if winner == home else home)
        return by_pair, eliminated

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
        # Official knockout structure (DAG + Annexe C) — the single source of truth, loaded once.
        self.spec = knockout.build_spec(knockout.load_rules())
        self.layout = knockout.occupancy_layout(self.spec)
        self._by_id = {m.id: m for m in self.spec.matches}
        # Static Round-of-32 slot identities, aligned with the occupancy participant-slot order, for
        # the bracket sub-labels (e.g. "Winners Group E" / "Best 3rd (vs Group E)").
        self._r32_tokens = [
            self._by_id[mid].a if side == "a" else self._by_id[mid].b
            for mid, side in self.layout.r32_slot_order
        ]

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
        """Simulate one group; return ordered standings as a list of dicts (1st..4th).

        Every scoreline is retained as an oriented ``(home, away, gh, ga)`` record so the official
        head-to-head tiebreakers can be built; final ordering is delegated to :func:`order_standings`.
        """
        teams = self.groups[g]
        stat = {t: {"pts": 0, "gd": 0, "gf": 0} for t in teams}
        results: list[tuple] = []
        for a, b, p, lam_a, lam_b in self.fixtures[g]:
            known = self._known.get(frozenset((a, b)))
            if known is not None:
                ga, gb = known["home_score"], known["away_score"]
                if known["home"] != a:  # stored orientation is reversed
                    ga, gb = gb, ga
            else:
                ga, gb = self._sample_score(p, lam_a, lam_b)
            results.append((a, b, ga, gb))
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
        return order_standings(standings, results, self.fifa_rank)

    def _knockout(self, a: str, b: str) -> str:
        p = self.adv[self.idx[a], self.idx[b]]
        return a if self.rng.random() < p else b

    def _resolve_knockout(self, a: str, b: str) -> str:
        """Winner of a knockout tie, honoring any completed real result before sampling.

        Priority: (1) the exact recorded winner if this pair has already met for real — this keeps
        upsets faithful; (2) if exactly one side has actually been knocked out (bracket drift paired
        them elsewhere), the other side advances; (3) otherwise sample the model as usual. A pair of
        two eliminated teams (e.g. the third-place match) falls to sampling — neither can advance.
        """
        forced = self._known_ko.get(frozenset((a, b)))
        if forced in (a, b):
            return forced
        a_out, b_out = a in self._eliminated, b in self._eliminated
        if a_out and not b_out:
            return b
        if b_out and not a_out:
            return a
        return self._knockout(a, b)

    def simulate_once(self) -> dict:
        """Play one full tournament; return the set of teams reaching each stage + the final.

        Plays the 12 groups, picks the eight best third-placed teams, resolves the official Annexe C
        third-place opponents from *which eight groups* qualified, then walks the fixed match DAG
        (M73→M104) once, resolving each slot token to a team and sampling the knockout result.
        """
        winners, runners, thirds = {}, {}, []
        for g in self.groups:
            standings = self._play_group(g)
            winners[g] = standings[0]["team"]
            runners[g] = standings[1]["team"]
            thirds.append(standings[2])

        best_thirds = select_best_thirds(thirds, self.best_thirds_count, self.fifa_rank)
        qualifying_groups = {t["group"] for t in best_thirds}
        annex = knockout.annex_assignment(self.spec.annex_matrix, qualifying_groups)
        third_of_group = {t["group"]: t["team"] for t in thirds}

        won: dict[str, str] = {}  # advance/loser token (W73 / L101) -> team
        winner_team: dict[str, str] = {}  # match id -> winner
        participants: dict[str, tuple[str, str]] = {}  # match id -> (team_a, team_b)

        def resolve(token: str) -> str:
            if token.startswith(knockout.ANNEX_PREFIX):
                third_slot = annex[token[len(knockout.ANNEX_PREFIX) :]]  # "1E" -> "3F"
                return third_of_group[third_slot[1:]]
            head = token[0]
            if head in ("W", "L"):
                return won[token]
            group = token[1:]
            if head == "1":
                return winners[group]
            if head == "2":
                return runners[group]
            return third_of_group[group]  # "3X" — a third-placed team referenced directly

        for m in self.spec.matches:
            a, b = resolve(m.a), resolve(m.b)
            participants[m.id] = (a, b)
            win = self._resolve_knockout(a, b)
            winner_team[m.id] = win
            if m.win_token:
                won[m.win_token] = win
            won[m.lose_token] = (
                b if win == a else a
            )  # loser (the third-place match consumes L101/L102)

        champion = winner_team[self.spec.final_id]
        rows = self.layout.rows_by_round

        reached = {s: [] for s in STAGES}
        # Teams that *reached* each stage: all 32 R32 participants, then each round's winners.
        reached["R32"] = (
            list(winners.values()) + list(runners.values()) + [t["team"] for t in best_thirds]
        )
        reached["R16"] = [winner_team[mid] for mid in rows["R32"]]
        reached["QF"] = [winner_team[mid] for mid in rows["R16"]]
        reached["SF"] = [winner_team[mid] for mid in rows["QF"]]
        reached["Final"] = [winner_team[mid] for mid in rows["SF"]]
        reached["Champion"] = [champion]

        self._last_participants = participants
        self._last_winner_team = winner_team
        self._last_r32_pairs = [participants[mid] for mid in rows["R32"]]
        self._last_final = tuple(sorted(participants[self.spec.final_id]))
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
        # that bracket participant slot. The slot geometry comes from the official-bracket layout
        # (:func:`core.knockout.occupancy_layout`): participant slot ``p`` of a round holds the winner
        # of the previous round's match at row ``p`` — so occupancy and advancement are one array read
        # a round apart, the identity :mod:`core.bracket` is built on.
        occ = {k: np.zeros((bracket_mod.PARTICIPANTS[k], n)) for k in bracket_mod.PARTICIPANTS}
        rows = self.layout.rows_by_round
        r32_slots = self.layout.r32_slot_order
        feed = [("R16", "R32"), ("QF", "R16"), ("SF", "QF"), ("Final", "SF")]

        for _ in range(n_sims):
            reached = self.simulate_once()
            for s in STAGES:
                for t in reached[s]:
                    counts[s][self.idx[t]] += 1
            parts = self._last_participants
            wt = self._last_winner_team
            # Round-of-32 participants, in bracket participant-slot order.
            r32_idx = [
                self.idx[parts[mid][0] if side == "a" else parts[mid][1]] for mid, side in r32_slots
            ]
            occ["R32"][np.arange(32), r32_idx] += 1
            # Each later round's slot p is the winner of the previous round's match at row p.
            for rk, prev in feed:
                w_idx = [self.idx[wt[mid]] for mid in rows[prev]]
                occ[rk][np.arange(len(w_idx)), w_idx] += 1
            occ["Champion"][0, self.idx[wt[self.spec.final_id]]] += 1
            fp = self._last_final
            final_pairs[fp] = final_pairs.get(fp, 0) + 1

        table = pd.DataFrame({"team": self.teams, "group": [self.group_of[t] for t in self.teams]})
        for s in STAGES:
            table[s] = counts[s] / n_sims
        table = table.sort_values("Champion", ascending=False).reset_index(drop=True)

        best_final = max(final_pairs.items(), key=lambda kv: kv[1])
        return SimulationResult(
            table=table,
            most_likely_final=best_final[0],
            most_likely_final_prob=best_final[1] / n_sims,
            n_sims=n_sims,
            bracket=bracket_mod.build_bracket(occ, self.teams, n_sims, self._r32_tokens),
        )
