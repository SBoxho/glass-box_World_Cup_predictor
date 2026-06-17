"""Guardrail — the per-team dossier re-projects the simulator/model outputs faithfully.

``core.team_page`` powers the fan-facing "Choose Your Team" page. It does no modelling of its own —
it slices an already-computed :class:`core.simulate.SimulationResult` and the inference state — so
the tests pin the contracts that matter: the field-relative ranking maths, the stage-probability
extraction, the *graceful fallbacks* when FIFA / squad data is absent (the brief's hard requirement),
the live group standings, the next-fixture selection, and that the plain-English read is grounded in
the numbers. A real (stub-driven) simulation is run once so the wiring is checked end-to-end.
"""

from __future__ import annotations

import types
from datetime import UTC, datetime

import pytest

from core import team_page as tp
from core.ingest import load_wc2026
from core.simulate import TournamentSimulator
from tests.test_simulate_format import StubPredictor


# --------------------------------------------------------------------------------------
# Fixtures — a hermetic simulation + a stand-in inference state
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def wc() -> dict:
    return load_wc2026()


@pytest.fixture(scope="module")
def result(wc):
    teams = tp.wc_teams(wc)
    sim = TournamentSimulator(StubPredictor(teams), wc, seed=0)
    return sim.run(n_sims=400, seed=0)


def make_state(wc, *, with_fifa=True, with_squad=False, fifa_subset=None):
    """A minimal stand-in for ``InferenceState`` — only the attributes team_page reads.

    Elo and (optionally) FIFA points / squad strength increase with the team's index, so the
    strongest team in the field is unambiguous and ranking assertions are deterministic.
    """
    teams = tp.wc_teams(wc)
    ratings = {t: 1400.0 + 5.0 * i for i, t in enumerate(teams)}
    team_form = {t: [(3, 2), (1, 0), (3, 1), (0, -2), (1, 0)] for t in teams}
    if with_fifa:
        keep = set(fifa_subset) if fifa_subset is not None else set(teams)
        fifa = {t: 1000.0 + 7.0 * i for i, t in enumerate(teams) if t in keep}
    else:
        fifa = {}
    state = types.SimpleNamespace(ratings=ratings, team_form=team_form, fifa_points=fifa)
    if with_squad:
        # Mirror the real InferenceState shape: team -> components dict (core.squads.METRICS),
        # of which bestxi_ovr is the headline overall the team page surfaces.
        state.squad_strength = {
            t: {
                "bestxi_ovr": 60.0 + 0.5 * i,
                "attack_ovr": 60.0 + 0.4 * i,
                "def_ovr": 60.0 + 0.3 * i,
                "depth_ovr": 58.0 + 0.4 * i,
                "star3_ovr": 62.0 + 0.5 * i,
            }
            for i, t in enumerate(teams)
        }
    return state


# --------------------------------------------------------------------------------------
# Ranking maths
# --------------------------------------------------------------------------------------
def test_rank_percentile_best_worst_and_ties():
    vals = [10.0, 20.0, 30.0, 40.0]
    assert tp._rank_percentile(40.0, vals) == (1, 4, 1.0)  # best
    assert tp._rank_percentile(10.0, vals) == (4, 4, 0.0)  # worst
    rank, n, pct = tp._rank_percentile(30.0, vals)
    assert (rank, n) == (2, 4) and pct == pytest.approx(2 / 3)
    # Ties share the better (lower) rank: two 5s both rank 1.
    assert tp._rank_percentile(5.0, [5.0, 5.0, 1.0])[0] == 1


def test_classify_thresholds():
    assert tp._classify(0.9) == "strength"
    assert tp._classify(0.5) == "neutral"
    assert tp._classify(0.1) == "weakness"


# --------------------------------------------------------------------------------------
# Stage probabilities
# --------------------------------------------------------------------------------------
def test_stage_probabilities_order_and_values(wc, result):
    team = result.table.iloc[0]["team"]  # the most likely champion in this run
    stages = tp.stage_probabilities(result, team)
    assert [s["key"] for s in stages] == tp.STAGES
    # Monotonically non-increasing (can't reach the final without the semis).
    ps = [s["p"] for s in stages]
    assert all(ps[i] >= ps[i + 1] - 1e-9 for i in range(len(ps) - 1))
    # And they match the table row exactly.
    row = result.table.loc[result.table["team"] == team].iloc[0]
    assert stages[0]["p"] == pytest.approx(float(row["R32"]))


def test_stage_probabilities_unknown_team(result):
    assert tp.stage_probabilities(result, "Wakanda") == []


def test_champion_field_max_is_table_max(result):
    assert tp.champion_field_max(result) == pytest.approx(float(result.table["Champion"].max()))


# --------------------------------------------------------------------------------------
# Factor breakdown — including the FIFA / squad graceful fallbacks
# --------------------------------------------------------------------------------------
def test_elo_factor_strength_and_weakness(wc):
    state = make_state(wc)
    teams = tp.wc_teams(wc)
    strong = {f["key"]: f for f in tp.factor_breakdown(state, wc, teams[-1])}
    weak = {f["key"]: f for f in tp.factor_breakdown(state, wc, teams[0])}
    assert strong["elo"]["kind"] == "strength" and strong["elo"]["rank"] == 1
    assert weak["elo"]["kind"] == "weakness" and weak["elo"]["rank"] == len(teams)


def test_squad_factor_is_na_without_squad_data(wc):
    state = make_state(wc, with_squad=False)
    squad = {f["key"]: f for f in tp.factor_breakdown(state, wc, "Brazil")}["squad"]
    assert squad["kind"] == "na"
    assert squad["value_str"] == "—"
    assert "isn't loaded" in squad["note"]  # honest fallback copy, not a fabricated rating


def test_squad_factor_activates_when_data_present(wc):
    state = make_state(wc, with_squad=True)
    squad = {f["key"]: f for f in tp.factor_breakdown(state, wc, tp.wc_teams(wc)[-1])}["squad"]
    assert squad["kind"] != "na"
    assert squad["value_str"].endswith("OVR")


def test_fifa_factor_is_na_when_feed_unavailable(wc):
    state = make_state(wc, with_fifa=False)
    fifa = {f["key"]: f for f in tp.factor_breakdown(state, wc, "Brazil")}["fifa"]
    assert fifa["kind"] == "na"


def test_fifa_factor_is_na_for_an_uncovered_team(wc):
    # Feed covers only the first two teams; a team outside it falls back gracefully.
    teams = tp.wc_teams(wc)
    state = make_state(wc, fifa_subset=teams[:2])
    fifa = {f["key"]: f for f in tp.factor_breakdown(state, wc, teams[-1])}["fifa"]
    assert fifa["kind"] == "na"


def test_factor_breakdown_order_leads_with_elo(wc):
    keys = [f["key"] for f in tp.factor_breakdown(make_state(wc), wc, "Brazil")]
    assert keys[0] == "elo"
    assert set(keys) == {"elo", "form", "gd", "fifa", "squad"}


# --------------------------------------------------------------------------------------
# Host context
# --------------------------------------------------------------------------------------
def test_host_context(wc):
    assert tp.host_context(wc, "Mexico") == {"is_host": True, "group": "A"}
    assert tp.host_context(wc, "Brazil") is None


# --------------------------------------------------------------------------------------
# Group standings + outlook
# --------------------------------------------------------------------------------------
def test_group_standings_none_until_a_match_is_played(wc):
    assert tp.group_standings(wc, "C", []) is None
    assert tp.group_standings(wc, "C", None) is None


def test_group_standings_orders_by_points(wc):
    known = [{"home": "Brazil", "away": "Morocco", "home_score": 3, "away_score": 0}]
    standings = tp.group_standings(wc, "C", known)
    assert standings is not None
    assert standings[0]["team"] == "Brazil"
    assert standings[0]["pts"] == 3 and standings[0]["gd"] == 3 and standings[0]["played"] == 1
    morocco = next(s for s in standings if s["team"] == "Morocco")
    assert morocco["pts"] == 0 and morocco["played"] == 1
    # Teams that haven't played yet are present with zero games.
    haiti = next(s for s in standings if s["team"] == "Haiti")
    assert haiti["played"] == 0


def test_group_outlook_structure(wc, result):
    state = make_state(wc)
    outlook = tp.group_outlook(wc, result, state, "Brazil")
    assert outlook["group"] == "C"
    assert len(outlook["rivals"]) == 4
    assert sum(r["is_self"] for r in outlook["rivals"]) == 1
    # advance_p equals the team's R32 probability from the table.
    row = result.table.loc[result.table["team"] == "Brazil"].iloc[0]
    assert outlook["advance_p"] == pytest.approx(float(row["R32"]))
    # strongest_rival is the highest-Elo team in the group other than Brazil.
    others = [t for t in wc["groups"]["C"] if t != "Brazil"]
    expected = max(others, key=lambda t: state.ratings[t])
    assert outlook["strongest_rival"] == expected


def test_group_outlook_unknown_team(wc, result):
    assert tp.group_outlook(wc, result, make_state(wc), "Wakanda") is None


# --------------------------------------------------------------------------------------
# Next fixture selection
# --------------------------------------------------------------------------------------
def test_next_team_fixture_picks_soonest_group_match():
    now = datetime(2026, 6, 11, tzinfo=UTC)
    past = {"team1": "Brazil", "team2": "Haiti", "kickoff_utc": "2026-06-01T22:00:00+00:00"}
    soon = {"team1": "Brazil", "team2": "Morocco", "kickoff_utc": "2026-06-13T22:00:00+00:00"}
    later = {"team1": "Scotland", "team2": "Brazil", "kickoff_utc": "2026-06-24T22:00:00+00:00"}
    knockout = {"team1": "2A", "team2": "2B", "kickoff_utc": "2026-06-28T19:00:00+00:00"}
    pick = tp.next_team_fixture([past, knockout, later, soon], "Brazil", now)
    assert pick is soon


def test_next_team_fixture_none_when_no_upcoming():
    now = datetime(2026, 7, 1, tzinfo=UTC)
    past = {"team1": "Brazil", "team2": "Morocco", "kickoff_utc": "2026-06-13T22:00:00+00:00"}
    assert tp.next_team_fixture([past], "Brazil", now) is None


# --------------------------------------------------------------------------------------
# "What needs to happen?" narrative
# --------------------------------------------------------------------------------------
def test_what_needs_to_happen_is_grounded(wc, result):
    state = make_state(wc)
    lines = tp.what_needs_to_happen(wc, result, state, "Brazil")
    assert lines
    assert "Group C" in lines[0]
    assert any("%" in ln for ln in lines)  # always surfaces a probability
    assert any("shot" in ln or "title path" in ln for ln in lines)  # the title/uncertainty line


def test_what_needs_to_happen_flags_host(wc, result):
    lines = tp.what_needs_to_happen(wc, result, make_state(wc), "Mexico")
    assert any("host" in ln.lower() for ln in lines)


def test_what_needs_to_happen_unknown_team(wc, result):
    assert tp.what_needs_to_happen(wc, result, make_state(wc), "Wakanda") == []


# --------------------------------------------------------------------------------------
# End-to-end dossier
# --------------------------------------------------------------------------------------
def test_build_dossier_unknown_team(wc, result):
    d = tp.build_dossier(
        wc, result, make_state(wc), [], "Wakanda", datetime(2026, 6, 11, tzinfo=UTC)
    )
    assert d["in_field"] is False


def test_build_dossier_full_shape(wc, result):
    now = datetime(2026, 6, 11, tzinfo=UTC)
    d = tp.build_dossier(wc, result, make_state(wc), [], "Brazil", now)
    assert d["in_field"] is True
    assert d["group"] == "C"
    assert len(d["stages"]) == 6
    assert len(d["factors"]) == 5
    assert d["what_needs_to_happen"]
    assert d["next_fixture"] is None  # no fixtures supplied
    assert d["n_sims"] == result.n_sims
