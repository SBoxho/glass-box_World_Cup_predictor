"""Bracket projection from the Monte Carlo simulation — per-slot occupancy + the story highlights.

The :class:`core.simulate.TournamentSimulator` answers "how often does each team reach each stage"
(a per-team table). A *bracket* needs more: for every position in the knockout tree it needs the
distribution over *which* team lands there — so a card can say "Brazil 62% to reach this slot" and
"who is most likely to advance from this match". This module turns the raw position-occupancy counts
the simulator accumulates into that render-ready structure, and derives the editorial highlights
(projected champion, Cinderella run, closest call) that make the bracket a story rather than a grid.

Pure and framework-free (numpy only): the simulator collects counts, this assembles them, and the
``app`` layer is a thin view. Everything here is unit-tested in ``tests/test_bracket.py``.

Bracket geometry (a perfect binary tree, the 2026 single-elimination phase)::

    R32: 16 matches / 32 participant slots   advance →  R16 slots
    R16:  8 matches / 16 slots                          QF  slots
    QF :  4 matches /  8 slots                          SF  slots
    SF :  2 matches /  4 slots                          Final slots
    Final: 1 match  /  2 slots                          Champion

Match ``m`` of a round owns participant slots ``2m`` (top) and ``2m+1`` (bottom); its winner is
participant slot ``m`` of the *next* round — so the next round's occupancy of slot ``m`` IS the
probability that team advances from match ``m``. That identity is the whole trick: occupancy and
advancement are the same array read one round apart.
"""

from __future__ import annotations

import numpy as np

# Knockout rounds that are *matches* (Champion is the outcome of the Final, not a round of matches).
ROUND_KEYS = ["R32", "R16", "QF", "SF", "Final"]
ROUND_TITLES = {
    "R32": "Round of 32",
    "R16": "Round of 16",
    "QF": "Quarter-finals",
    "SF": "Semi-finals",
    "Final": "Final",
}
# participant-slot count per round/outcome (drives the occupancy array shapes)
PARTICIPANTS = {"R32": 32, "R16": 16, "QF": 8, "SF": 4, "Final": 2, "Champion": 1}
# the round whose participant occupancy gives match-m's advancement distribution
_NEXT = {"R32": "R16", "R16": "QF", "QF": "SF", "SF": "Final", "Final": "Champion"}
# number of matches per round
N_MATCHES = {"R32": 16, "R16": 8, "QF": 4, "SF": 2, "Final": 1}


def token_label(token: str | None) -> str | None:
    """Human label for a Round-of-32 slot token, or ``None`` if it isn't a recognised slot.

    Two token vocabularies are accepted so the label layer is agnostic to where the slots come from:

    * the **official** rules-file tokens (:mod:`core.knockout`) — ``1X`` (group winner), ``2X``
      (runner-up), ``3X`` (third-placed), and ``ANNEX_C_FOR_1X`` (the Annexe C best-third opponent of
      group-``X``'s winner); and
    * the legacy ``W_x`` / ``R_x`` / ``T_n`` template tokens (kept for backward compatibility).

    Each gives a Round-of-32 slot a stable identity, so a card can show *what* the slot is even before
    *who* fills it.
    """
    if not token:
        return None
    if token.startswith("ANNEX_C_FOR_"):
        group = token[len("ANNEX_C_FOR_") + 1 :]  # "ANNEX_C_FOR_1E" -> "E"
        return f"Best 3rd (vs Group {group})"
    if token[0] in "123" and token[1:].isalpha():  # official 1X / 2X / 3X
        kind, group = token[0], token[1:]
        if kind == "1":
            return f"Winners Group {group}"
        if kind == "2":
            return f"Runners-up Group {group}"
        return f"3rd place Group {group}"
    if "_" in token:  # legacy W_x / R_x / T_n
        kind, rest = token.split("_", 1)
        if kind == "W":
            return f"Winners Group {rest}"
        if kind == "R":
            return f"Runners-up Group {rest}"
        if kind == "T":
            return f"3rd place #{rest}"
    return None


def coherent_modal_assignment(round_occ: np.ndarray) -> list[int | None]:
    """One team per slot for a round: a greedy maximum-weight assignment of teams to participant slots.

    ``round_occ`` is a ``(slots, teams)`` occupancy-count matrix. We form every ``(count, slot,
    team)`` triple, take them in descending count, and give each team to its highest-count *still
    free* slot — so no team can lead two slots of the same round. That is the fix for a dominant team
    rendering as both a group's plurality *winner* (its ``1C`` slot) **and** its plurality
    *runner-up* (its ``2C`` slot): independent per-slot argmax double-counts it; this does not.

    Greedy-by-descending-count (rather than an optimal Hungarian assignment) is deterministic,
    ``O(slots·teams·log)``, and near-optimal on the sharply peaked occupancy matrices the simulator
    produces — a drop-in upgrade to Hungarian is possible if ever needed. Returns a length-``slots``
    list of the assigned team index, or ``None`` for a slot that received no team.
    """
    n_slots, n_teams = round_occ.shape
    triples = [
        (round_occ[s, j], s, j)
        for s in range(n_slots)
        for j in range(n_teams)
        if round_occ[s, j] > 0
    ]
    triples.sort(
        key=lambda t: (-t[0], t[1], t[2])
    )  # deterministic: count desc, then slot, then team
    slot_team: list[int | None] = [None] * n_slots
    used: set[int] = set()
    for _count, s, j in triples:
        if slot_team[s] is None and j not in used:
            slot_team[s] = j
            used.add(j)
    return slot_team


def _dist(counts_row: np.ndarray, teams: list[str], n_sims: int, top_k: int, eps: float) -> list:
    """A slot's occupancy distribution as ``[{"team", "p"}, …]`` — top-k teams, p ≥ eps, desc."""
    if n_sims <= 0:
        return []
    order = np.argsort(counts_row)[::-1][:top_k]
    out = []
    for j in order:
        p = float(counts_row[j]) / n_sims
        if p < eps:
            break
        out.append({"team": teams[int(j)], "p": p})
    return out


def _lead_dist(
    counts_row: np.ndarray,
    teams: list[str],
    n_sims: int,
    top_k: int,
    eps: float,
    lead: int | None,
) -> list:
    """A slot's distribution (see :func:`_dist`) with the coherent lead team forced to element 0.

    Only *which* team leads can change — every probability ``p`` is the team's own occupancy share,
    left untouched (a glass-box honesty guarantee). Moving one element to the front keeps the tail
    (index ≥ 1) descending while the list as a whole may not be — the deliberate, documented trade
    for a coherent one-team-per-slot projection. ``lead`` is the team index chosen by
    :func:`coherent_modal_assignment` (``None`` leaves the raw argmax order in place).
    """
    dist = _dist(counts_row, teams, n_sims, top_k, eps)
    if lead is None:
        return dist
    lead_team = teams[lead]
    for i, entry in enumerate(dist):
        if entry["team"] == lead_team:
            if i:
                dist.insert(0, dist.pop(i))
            return dist
    # The coherent lead fell outside this slot's top-k / eps list — prepend it at its true share.
    p = float(counts_row[lead]) / n_sims if n_sims > 0 else 0.0
    if p > 0:
        dist.insert(0, {"team": lead_team, "p": p})
        del dist[top_k:]
    return dist


def build_bracket(
    occ: dict[str, np.ndarray],
    teams: list[str],
    n_sims: int,
    r32_tokens: list[str | None] | None = None,
    *,
    top_k: int = 4,
    eps: float = 0.004,
) -> dict:
    """Assemble the render-ready bracket from per-round position-occupancy count arrays.

    ``occ[round]`` has shape ``(PARTICIPANTS[round], n_teams)``: row *p* counts how many of the
    ``n_sims`` tournaments placed each team in participant slot *p* of that round (bracket order).
    ``r32_tokens`` (optional, length 32) gives each Round-of-32 slot its official identity for a
    sub-label. Returns ``{"n_sims", "rounds": [...], "champion": [...]}`` where each round holds its
    matches and each match its ``top``/``bottom`` occupancy lists and ``advance`` distribution.

    Each round is projected **coherently** — :func:`coherent_modal_assignment` picks a single leading
    team per participant slot so no team heads two slots of the same round (an aggregation artifact a
    dominant side would otherwise produce). Only the *leader* of each slot is pinned; the rest of the
    per-slot distribution, and every probability, is the honest raw occupancy. A match's ``advance``
    is reordered by the *next* round's slot leader, so the favourite shown advancing matches the team
    shown occupying that slot one round on.
    """
    assign = {rk: coherent_modal_assignment(occ[rk]) for rk in PARTICIPANTS}
    rounds = []
    for rk in ROUND_KEYS:
        nxt = occ[_NEXT[rk]]
        cur = occ[rk]
        a_cur, a_nxt = assign[rk], assign[_NEXT[rk]]
        matches = []
        for m in range(N_MATCHES[rk]):
            top = _lead_dist(cur[2 * m], teams, n_sims, top_k, eps, a_cur[2 * m])
            bottom = _lead_dist(cur[2 * m + 1], teams, n_sims, top_k, eps, a_cur[2 * m + 1])
            advance = _lead_dist(nxt[m], teams, n_sims, top_k, eps, a_nxt[m])
            match = {"round": rk, "index": m, "top": top, "bottom": bottom, "advance": advance}
            if rk == "R32" and r32_tokens is not None:
                match["top_label"] = token_label(r32_tokens[2 * m])
                match["bottom_label"] = token_label(r32_tokens[2 * m + 1])
            matches.append(match)
        rounds.append({"key": rk, "title": ROUND_TITLES[rk], "matches": matches})
    champion = _lead_dist(occ["Champion"][0], teams, n_sims, top_k, eps, assign["Champion"][0])
    return {"n_sims": n_sims, "rounds": rounds, "champion": champion}


# --------------------------------------------------------------------------------------
# Editorial highlights — the four call-outs the brief asks the bracket to surface
# --------------------------------------------------------------------------------------
def _stage_reach(table, team: str) -> dict[str, float]:
    """Per-stage reach probabilities for a team, read from the simulator's aggregate table."""
    row = table.loc[table["team"] == team]
    if row.empty:
        return {}
    r = row.iloc[0]
    return {s: float(r[s]) for s in ["R32", "R16", "QF", "SF", "Final", "Champion"] if s in r}


def _elo_ranks(teams: list[str], ratings: dict[str, float] | None) -> dict[str, int]:
    """Map each team to its 1-based strength rank by Elo (unknown ratings sort last)."""
    if not ratings:
        return {t: i + 1 for i, t in enumerate(teams)}
    ordered = sorted(teams, key=lambda t: ratings.get(t, float("-inf")), reverse=True)
    return {t: i + 1 for i, t in enumerate(ordered)}


def most_uncertain_match(bracket: dict, *, min_determined: float = 0.22) -> dict | None:
    """The bracket's closest call: a near-coin-flip between two reasonably-determined participants.

    For each match we take the two modal participants (top/bottom slot leaders) and read their
    advancement probabilities; ``closeness`` is 1 for a perfect 50/50. ``min_determined`` (how likely
    this exact pairing actually occurs) is a *gate* — it keeps out matches whose very participants are
    unknown — but does not otherwise tip the ranking, so a genuine coin-flip wins on closeness rather
    than on how settled the bracket is around it. Later rounds break ties (a semi-final coin-flip is
    a better story than a Round-of-32 one). Returns ``None`` if nothing qualifies.
    """
    depth_weight = {"R32": 1.0, "R16": 1.15, "QF": 1.3, "SF": 1.5, "Final": 1.7}
    best, best_score = None, -1.0
    for rnd in bracket["rounds"]:
        for match in rnd["matches"]:
            if not match["top"] or not match["bottom"]:
                continue
            a, b = match["top"][0], match["bottom"][0]
            if a["team"] == b["team"]:
                continue
            adv = {d["team"]: d["p"] for d in match["advance"]}
            pa, pb = adv.get(a["team"], 0.0), adv.get(b["team"], 0.0)
            if pa + pb <= 0:
                continue
            determined = min(a["p"], b["p"])
            if determined < min_determined:
                continue
            closeness = 1.0 - abs(pa - pb) / (pa + pb)
            score = closeness * depth_weight[match["round"]]
            if score > best_score:
                best_score = score
                total = pa + pb
                best = {
                    "round": match["round"],
                    "round_title": ROUND_TITLES[match["round"]],
                    "index": match["index"],
                    "team_a": a["team"],
                    "team_b": b["team"],
                    "p_a": pa / total,  # conditional P(win | this pairing)
                    "p_b": pb / total,
                    "closeness": closeness,
                }
    return best


def biggest_upset(table, ratings: dict[str, float] | None, *, seed_cutoff: int = 8) -> dict | None:
    """The Cinderella: the strongest deep-run odds among teams *outside* the Elo top ``seed_cutoff``.

    We rank the field by Elo, then among the non-seeds pick the team most likely to reach the
    semi-finals (falling back to the quarters as the headline stage if no underdog clears a small
    floor). This surfaces a genuine underdog run — a low-ranked side the model nonetheless gives a
    real chance of going deep — rather than just the overall favourite. ``None`` if the field is
    empty or no underdog has a meaningful deep run.
    """
    teams = list(table["team"])
    if not teams:
        return None
    ranks = _elo_ranks(teams, ratings)
    candidates = [t for t in teams if ranks[t] > seed_cutoff]
    if not candidates:
        return None
    reach = {t: _stage_reach(table, t) for t in candidates}
    best = max(candidates, key=lambda t: reach[t].get("SF", 0.0))
    p_sf = reach[best].get("SF", 0.0)
    p_qf = reach[best].get("QF", 0.0)
    if p_qf < 0.02:  # not even a flicker of a deep run — don't manufacture a story
        return None
    stage, p = ("SF", p_sf) if p_sf >= 0.05 else ("QF", p_qf)
    return {
        "team": best,
        "stage": stage,
        "stage_title": {"QF": "quarter-finals", "SF": "semi-finals"}[stage],
        "p": p,
        "p_qf": p_qf,
        "p_sf": p_sf,
        "elo_rank": ranks[best],
    }


def team_path(bracket: dict, team: str | None, *, min_p: float = 0.06) -> dict[str, dict]:
    """Where ``team`` most likely sits in each round → ``{round_key: {index, side, p}}``.

    For each round, find the match/side where the team has its highest occupancy; include it only
    when that probability clears ``min_p`` (so we trace a believable path, not statistical noise).
    Used to highlight a team's road through the bracket — the projected champion's, the Cinderella's,
    or the user-selected team's. Empty dict when ``team`` is falsy or never appears.
    """
    if not team:
        return {}
    path: dict[str, dict] = {}
    for rnd in bracket["rounds"]:
        best = None
        for match in rnd["matches"]:
            for side in ("top", "bottom"):
                for d in match[side]:
                    if d["team"] == team and (best is None or d["p"] > best["p"]):
                        best = {"index": match["index"], "side": side, "p": d["p"]}
        if best is not None and best["p"] >= min_p:
            path[rnd["key"]] = best
    return path


def derive_highlights(result, ratings: dict[str, float] | None) -> dict:
    """Bundle the editorial highlights for a finished simulation (``result`` has ``.table``/``.bracket``).

    Returns ``{"champion", "upset", "toss_up"}`` — the projected champion (name + title odds), the
    Cinderella run, and the closest call. Any element may be ``None`` if the data doesn't support it
    (e.g. no underdog with a deep run). The user-selected-team path is computed separately, on demand.
    """
    bracket = getattr(result, "bracket", None)
    champion = None
    if bracket and bracket.get("champion"):
        champion = dict(bracket["champion"][0])
    return {
        "champion": champion,
        "upset": biggest_upset(result.table, ratings),
        "toss_up": most_uncertain_match(bracket) if bracket else None,
    }
