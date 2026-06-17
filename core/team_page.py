"""Per-team "dossier" — everything the fan-facing **Choose Your Team** page needs, assembled
from the *existing* model and simulator outputs (no new modelling).

The Tournament Simulator already produces a :class:`core.simulate.SimulationResult` (a per-team
table of stage-reach probabilities + a per-slot bracket projection), and the inference
:class:`core.features.InferenceState` already holds each team's current strength inputs (Elo, FIFA
points, recent form). This module re-projects those onto a *single chosen team* so the UI can show:

* the team's probability of reaching each stage (Round of 32 → Champion);
* its group outlook — model advance odds for all four sides, plus live standings if any group
  matches have been played;
* a **field-relative factor breakdown** — where the team ranks among the 48 on each input the model
  actually consumes (Elo, recent form, recent scoring, FIFA points, squad strength). This is the
  honest substance behind "strengths & weaknesses": it is the model's own inputs, ranked against the
  field — never invented tactical claims;
* the team's most-likely road through the bracket (delegated to :func:`core.bracket.team_path`);
* the next group fixture to predict; and
* a plain-English "what needs to happen?" read, grounded in the advance odds, the toughest group
  rival, and the title-odds spread (so the model's uncertainty stays visible).

Pure and framework-free (the ``core`` rule): no Streamlit, no model *calls* — it only reads
already-computed objects. Squad strength and FIFA points degrade gracefully when their data is
absent (see :func:`factor_breakdown`), which is exactly the case on a build without the EA FC squad
feature: the factor is reported as *limited coverage* rather than guessed. Unit-tested in
``tests/test_team_page.py``.
"""

from __future__ import annotations

from core import bracket as cb
from core import config, fixtures
from core.features import _ppg_gd

# Stage keys (the simulator's columns) and their fan-facing labels.
STAGES = ["R32", "R16", "QF", "SF", "Final", "Champion"]
STAGE_LABELS = {
    "R32": "Round of 32",
    "R16": "Round of 16",
    "QF": "Quarter-finals",
    "SF": "Semi-finals",
    "Final": "Final",
    "Champion": "Champion",
}

# Percentile thresholds for calling a factor a strength / weakness (roughly the top and bottom third
# of the field). Between the two it reads as "around the field average".
_STRENGTH_PCT = 0.66
_WEAKNESS_PCT = 0.34


# --------------------------------------------------------------------------------------
# Field helpers
# --------------------------------------------------------------------------------------
def wc_teams(wc: dict) -> list[str]:
    """The 48 drawn teams (flattened from the group dict), in group order."""
    return [t for teams in wc["groups"].values() for t in teams]


def group_of(wc: dict, team: str) -> str | None:
    """The group letter the team is drawn into, or ``None`` if it is not in the field."""
    for g, teams in wc["groups"].items():
        if team in teams:
            return g
    return None


def is_host(wc: dict, team: str) -> bool:
    """True for a 2026 host (United States / Mexico / Canada) — they play group games at home."""
    return team in (wc.get("host_groups") or {}) or team in (wc.get("hosts") or [])


def _rank_percentile(value: float, values: list[float]) -> tuple[int, int, float]:
    """1-based rank (1 = best/highest) and percentile of ``value`` within ``values``.

    Percentile is the share of the *other* teams this value sits at or above, in ``[0, 1]`` — so the
    top team is ``1.0`` and the bottom team ``0.0``. Ties share the better rank.
    """
    n = len(values)
    if n == 0:
        return 0, 0, 0.0
    rank = 1 + sum(1 for v in values if v > value)
    pct = 1.0 if n == 1 else (n - rank) / (n - 1)
    return rank, n, max(0.0, min(1.0, pct))


def _classify(pct: float) -> str:
    if pct >= _STRENGTH_PCT:
        return "strength"
    if pct <= _WEAKNESS_PCT:
        return "weakness"
    return "neutral"


# --------------------------------------------------------------------------------------
# Stage probabilities (read straight off the simulator's aggregate table)
# --------------------------------------------------------------------------------------
def stage_probabilities(result, team: str) -> list[dict]:
    """``[{"key", "label", "p"}, …]`` for Round-of-32 → Champion, or ``[]`` if the team is absent.

    Reads the simulator's per-team table directly — ``R32`` is the probability of getting out of the
    group (top two or a best-third slot), and each later column the probability of reaching that
    round. The numbers *are* the uncertainty; the UI surfaces them as-is.
    """
    row = result.table.loc[result.table["team"] == team]
    if row.empty:
        return []
    r = row.iloc[0]
    return [{"key": s, "label": STAGE_LABELS[s], "p": float(r[s])} for s in STAGES if s in r]


def champion_field_max(result) -> float:
    """The single highest title probability in the field — the favourite's odds.

    Used to frame a team's own title odds honestly: in a 48-team field even the favourite is usually
    well under a coin flip, so showing this alongside a team's number keeps the spread visible.
    """
    if "Champion" not in result.table.columns or result.table.empty:
        return 0.0
    return float(result.table["Champion"].max())


# --------------------------------------------------------------------------------------
# Factor breakdown — the model's own inputs, ranked against the field
# --------------------------------------------------------------------------------------
def recent_form(state, team: str) -> tuple[float, float]:
    """The team's current ``(points-per-game, avg goal difference)`` over the model's form window.

    Reuses the exact aggregation the feature pipeline uses (:func:`core.features._ppg_gd`), so these
    are the same numbers the model is fed — not a parallel re-derivation that could drift.
    """
    return _ppg_gd(state.team_form.get(team, []))


def squad_value(state, team: str) -> float | None:
    """The team's overall squad strength (best-XI mean OVR), or ``None`` when squad data is absent.

    Reads the ``squad_strength`` mapping the inference state carries when the EA FC squad feature is
    loaded (``core.features.InferenceState``): each team maps to a *components* dict keyed by
    :data:`core.squads.METRICS`, of which ``bestxi_ovr`` is the headline overall. A plain-float
    mapping is also accepted for forward-compatibility. On a build without squad data — or a team the
    snapshot doesn't cover — this returns ``None`` and the factor reports *limited coverage* instead
    of a fabricated rating.
    """
    table = getattr(state, "squad_strength", None)
    if not isinstance(table, dict):
        return None
    comp = table.get(team)
    if isinstance(comp, dict):  # branch shape: per-team component dict (bestxi_ovr, attack_ovr, …)
        val = comp.get("bestxi_ovr")
        return float(val) if val is not None else None
    if comp is not None:  # forward-compat: a plain {team: overall} mapping
        return float(comp)
    return None


def _factor(key, label, value, field_values, *, value_str, available=True, note="") -> dict:
    """Assemble one factor row: the team's value, its field rank/percentile, and a classification."""
    if not available or value is None:
        return {
            "key": key,
            "label": label,
            "value": value,
            "value_str": "—",
            "rank": None,
            "n": len(field_values),
            "percentile": None,
            "kind": "na",
            "note": note,
        }
    rank, n, pct = _rank_percentile(value, field_values)
    return {
        "key": key,
        "label": label,
        "value": value,
        "value_str": value_str,
        "rank": rank,
        "n": n,
        "percentile": pct,
        "kind": _classify(pct),
        "note": note,
    }


def factor_breakdown(state, wc: dict, team: str) -> list[dict]:
    """Where ``team`` ranks among the 48 on each model input — ordered by the model's reliance.

    Each entry is a dict with the team's ``value``, its 1-based ``rank`` of ``n`` and ``percentile``,
    and a ``kind`` of ``"strength"`` / ``"neutral"`` / ``"weakness"`` / ``"na"``. The order (Elo
    first) reflects the honest finding that Elo is the dominant signal and FIFA points / squad
    strength add little measured lift — so the UI leads with what actually moves the model.

    Factors whose data is missing for this build (squad strength on ``main``; FIFA points if the
    ranking feed is unavailable) come back as ``kind="na"`` with a plain-English ``note`` — the
    graceful fallback the brief asks for, never an invented number.
    """
    field = wc_teams(wc)

    # Elo — always present (every WC side has match history); the model's dominant input.
    elos = [state.ratings.get(t, config.ELO_BASE) for t in field]
    team_elo = state.ratings.get(team, config.ELO_BASE)
    elo_factor = _factor(
        "elo",
        "Elo rating",
        team_elo,
        elos,
        value_str=f"{team_elo:.0f}",
        note="Chess-style strength rating — the single biggest driver of the model.",
    )

    # Recent form (points per game) and recent scoring (avg goal difference) over the form window.
    forms = [_ppg_gd(state.team_form.get(t, []))[0] for t in field]
    gds = [_ppg_gd(state.team_form.get(t, []))[1] for t in field]
    ppg, gd = recent_form(state, team)
    form_factor = _factor(
        "form",
        "Recent form",
        ppg,
        forms,
        value_str=f"{ppg:.2f} pts/game",
        note=f"Points per game across the last {config.FORM_WINDOW} matches.",
    )
    gd_factor = _factor(
        "gd",
        "Recent scoring",
        gd,
        gds,
        value_str=f"{gd:+.2f} GD/game",
        note=f"Average goal difference across the last {config.FORM_WINDOW} matches.",
    )

    # FIFA points — present only for teams the ranking feed covers; otherwise a graceful fallback.
    fifa = getattr(state, "fifa_points", {}) or {}
    field_fifa = [fifa[t] for t in field if t in fifa]
    have_fifa = team in fifa and len(field_fifa) >= 2
    fifa_factor = _factor(
        "fifa",
        "FIFA ranking points",
        fifa.get(team) if have_fifa else None,
        field_fifa,
        value_str=f"{fifa.get(team, 0):.0f} pts" if have_fifa else "—",
        available=have_fifa,
        note=(
            "Official FIFA points (a secondary signal — closely tracks Elo, so it adds little lift)."
            if have_fifa
            else "No current FIFA-ranking points on file for this team — Elo and form carry the read."
        ),
    )

    # Squad strength (EA FC) — absent on builds without the squad feature; honest fallback copy.
    squad = squad_value(state, team)
    field_squad = [v for v in (squad_value(state, t) for t in field) if v is not None]
    have_squad = squad is not None and len(field_squad) >= 2
    squad_factor = _factor(
        "squad",
        "Squad strength (EA FC)",
        squad if have_squad else None,
        field_squad,
        value_str=f"{squad:.0f} OVR" if have_squad else "—",
        available=have_squad,
        note=(
            "Aggregate squad rating from EA FC player data."
            if have_squad
            else "Squad-strength (EA FC) data isn't loaded in this build — not factored into the read."
        ),
    )

    return [elo_factor, form_factor, gd_factor, fifa_factor, squad_factor]


def host_context(wc: dict, team: str) -> dict | None:
    """Host-nation context for the chosen team, or ``None`` for the 45 non-hosts.

    A 2026 host plays all three group matches at home — the only non-neutral games in its run, and a
    genuine edge the model encodes via ``is_host_home``. Returned so the UI can show it *where
    relevant* rather than as a field ranking that would be meaningless for everyone else.
    """
    if not is_host(wc, team):
        return None
    return {"is_host": True, "group": (wc.get("host_groups") or {}).get(team) or group_of(wc, team)}


# --------------------------------------------------------------------------------------
# Group outlook
# --------------------------------------------------------------------------------------
def group_standings(wc: dict, group: str, known_results: list[dict] | None) -> list[dict] | None:
    """Live group standings from any *played* group matches, or ``None`` if none are in yet.

    ``known_results`` is the simulator's locked-results schema (``{home, away, home_score,
    away_score}``); team names are normalized so committed/live spellings join. Returns the four
    sides ordered by points → goal difference → goals for, each with games ``played`` — the same
    ordering the simulator uses — or ``None`` before any group game has been played.
    """
    teams = wc["groups"].get(group, [])
    stat = {t: {"team": t, "pts": 0, "gd": 0, "gf": 0, "played": 0} for t in teams}
    any_played = False
    for r in known_results or []:
        h = config.normalize_team(r.get("home"))
        a = config.normalize_team(r.get("away"))
        hs, as_ = r.get("home_score"), r.get("away_score")
        if h not in stat or a not in stat or hs is None or as_ is None:
            continue
        any_played = True
        hs, as_ = int(hs), int(as_)
        stat[h]["gf"] += hs
        stat[a]["gf"] += as_
        stat[h]["gd"] += hs - as_
        stat[a]["gd"] += as_ - hs
        stat[h]["played"] += 1
        stat[a]["played"] += 1
        if hs > as_:
            stat[h]["pts"] += 3
        elif hs == as_:
            stat[h]["pts"] += 1
            stat[a]["pts"] += 1
        else:
            stat[a]["pts"] += 3
    if not any_played:
        return None
    return sorted(stat.values(), key=lambda s: (s["pts"], s["gd"], s["gf"]), reverse=True)


def group_outlook(wc: dict, result, state, team: str, known_results=None) -> dict | None:
    """The chosen team's group picture: model advance odds for all four sides + live standings.

    Returns ``None`` if the team is not in the field. ``rivals`` lists the four sides (the chosen
    team flagged ``is_self``) with their Elo and model advance probability, ordered by advance odds.
    ``strongest_rival`` is the toughest opponent by Elo. ``standings`` is the live table when any
    group game has been played, else ``None``.
    """
    g = group_of(wc, team)
    if g is None:
        return None
    reach = {row["team"]: row for _, row in result.table.iterrows()}
    rivals = []
    for t in wc["groups"][g]:
        adv = float(reach[t]["R32"]) if t in reach and "R32" in result.table.columns else 0.0
        rivals.append(
            {
                "team": t,
                "elo": state.ratings.get(t, config.ELO_BASE),
                "advance_p": adv,
                "is_self": t == team,
            }
        )
    rivals.sort(key=lambda r: r["advance_p"], reverse=True)
    others = [r for r in rivals if not r["is_self"]]
    strongest = max(others, key=lambda r: r["elo"])["team"] if others else None
    own = next((r for r in rivals if r["is_self"]), None)
    return {
        "group": g,
        "advance_p": own["advance_p"] if own else 0.0,
        "rivals": rivals,
        "strongest_rival": strongest,
        "standings": group_standings(wc, g, known_results),
    }


# --------------------------------------------------------------------------------------
# Next fixture + bracket path
# --------------------------------------------------------------------------------------
def next_team_fixture(fixtures_list: list[dict], team: str, now) -> dict | None:
    """The chosen team's soonest upcoming fixture (real team names), or ``None``.

    Only fixtures with the team named on one side and a future, timed kick-off qualify — so this
    picks up the next *group* match (knockout slots are still placeholders like ``"2A"`` and are
    skipped, since the opponent isn't known yet). Mirrors :func:`core.fixtures.next_match`'s
    selection but filtered to one team.
    """
    cand = [
        fx
        for fx in fixtures_list
        if team in (fx.get("team1"), fx.get("team2"))
        and fixtures.match_status(fx, now) == "upcoming"
        and fixtures.kickoff_datetime(fx) is not None
    ]
    if not cand:
        return None
    return min(cand, key=lambda fx: fixtures.kickoff_datetime(fx))


def bracket_path(result, team: str) -> dict:
    """The team's most-likely position in each knockout round (delegates to the bracket projection).

    ``{round_key: {"index", "side", "p"}}`` for the rounds where the team has a believable presence
    — the same trace the bracket view highlights. Empty if the simulation had no bracket projection.
    """
    bracket = getattr(result, "bracket", None)
    if not bracket:
        return {}
    return cb.team_path(bracket, team)


# --------------------------------------------------------------------------------------
# "What needs to happen?" — plain English, grounded in the numbers above
# --------------------------------------------------------------------------------------
def what_needs_to_happen(
    wc: dict, result, state, team: str, outlook: dict | None = None
) -> list[str]:
    """A short, honest, data-grounded read of what the chosen team needs — no betting language.

    Each line is tied to a number the page already shows: the group advance odds, the toughest group
    rival by Elo, the drop-off into the Round of 16, and the title-odds spread (with the field
    favourite, so the model's uncertainty stays front and centre). Returns ``[]`` if the team is not
    in the field.
    """
    probs = {s["key"]: s["p"] for s in stage_probabilities(result, team)}
    if not probs:
        return []
    outlook = outlook or group_outlook(wc, result, state, team)
    g = outlook["group"] if outlook else group_of(wc, team)
    advance = probs.get("R32", 0.0)
    r16 = probs.get("R16", 0.0)
    champ = probs.get("Champion", 0.0)
    fav = champion_field_max(result)

    lines: list[str] = []
    lines.append(
        f"**{team}** are in **Group {g}**. To reach the knockouts they need a top-two finish — or "
        f"one of the eight best third-place spots. The model gives them **{advance:.0%}** to get "
        f"out of the group."
    )

    strongest = outlook.get("strongest_rival") if outlook else None
    if strongest:
        team_elo = state.ratings.get(team, config.ELO_BASE)
        riv_elo = state.ratings.get(strongest, config.ELO_BASE)
        rel = "above" if team_elo >= riv_elo else "below"
        lines.append(
            f"Their toughest group rival on the model's numbers is **{strongest}** "
            f"({riv_elo:.0f} Elo, with {team} {abs(team_elo - riv_elo):.0f} {rel})."
        )

    if advance >= 0.10:
        lines.append(
            f"The Round of 32 is the real test: even after getting through the group, only "
            f"**{r16:.0%}** of {team}'s simulated runs reach the last 16 — most ends come early in "
            f"the knockouts."
        )
    elif advance > 0:
        lines.append(
            f"Just escaping the group would be the story — the model has {team} advancing in only "
            f"{advance:.0%} of runs, so every group point matters."
        )

    if is_host(wc, team):
        lines.append(
            f"As a host, **{team}** play all three group games at home — the only non-neutral matches "
            f"in their tournament, and a real edge the model factors in."
        )

    if champ >= 0.005:
        if fav > 0 and champ >= fav - 1e-9:
            tail = "the model's outright favourite — though in a 48-team field that still means most simulated tournaments end with someone else lifting it."
        else:
            tail = (
                f"a long shot — for scale, even the field favourite tops out around **{fav:.0%}**, so "
                f"the title is wide open."
            )
        lines.append(f"Going all the way is a **{champ:.1%}** shot: {tail}")
    else:
        lines.append(
            f"The model doesn't give {team} a realistic title path this time — the value here is "
            f"seeing how far a deep run *could* go, and what it would take."
        )
    return lines


# --------------------------------------------------------------------------------------
# One-call assembly (everything the page needs except the live model prediction)
# --------------------------------------------------------------------------------------
def build_dossier(
    wc: dict, result, state, fixtures_list: list[dict], team: str, now, known_results=None
) -> dict:
    """Bundle the full team dossier for the UI in one call (model-call-free).

    The next-match *prediction* is intentionally left to the app layer (it needs the model), so this
    stays pure and testable; everything else — stage odds, group outlook, factor breakdown, bracket
    path, the next fixture to predict, and the narrative — is assembled here from already-computed
    objects. Returns ``in_field=False`` (with the rest empty) if the team isn't one of the 48.
    """
    if group_of(wc, team) is None:
        return {"team": team, "in_field": False}
    outlook = group_outlook(wc, result, state, team, known_results)
    return {
        "team": team,
        "in_field": True,
        "group": outlook["group"] if outlook else None,
        "stages": stage_probabilities(result, team),
        "champion_field_max": champion_field_max(result),
        "outlook": outlook,
        "factors": factor_breakdown(state, wc, team),
        "host": host_context(wc, team),
        "path": bracket_path(result, team),
        "next_fixture": next_team_fixture(fixtures_list, team, now),
        "what_needs_to_happen": what_needs_to_happen(wc, result, state, team, outlook),
        "n_sims": getattr(result, "n_sims", None),
    }
