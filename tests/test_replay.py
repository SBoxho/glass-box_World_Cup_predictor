"""Guardrail — the post-tournament replay has no look-ahead.

The Simulate tab's replay re-runs the finished 2026 World Cup "as the model saw it on the eve of the
opener" and then grades itself against reality. Agreement with reality only means something if no
tournament result leaks in, so this pins down every input path:

* the tournament definition carries no locked group/knockout result;
* team strength (Elo / form / head-to-head) and the FIFA ranking are *invariant* to anything dated
  on or after the cutoff — fabricated post-cutoff data must change nothing;
* the replay simulation is identical whatever the real results were;
* the committed replay model was trained strictly before the opener;
* the grading helper reconstructs each team's real furthest stage from bare knockout results.
"""

from __future__ import annotations

import pandas as pd
import pytest
from test_simulate_format import StubPredictor

from core import config, model, replay, simulate
from core.features import build_inference_state
from core.ingest import load_wc2026


def _teams(wc):
    return [t for grp in wc["groups"].values() for t in grp]


# --------------------------------------------------------------------------------------
# Inputs are cut at the opener
# --------------------------------------------------------------------------------------
def test_cutoff_is_the_opening_day():
    assert (
        replay.cutoff_from_wc(load_wc2026()) == replay.REPLAY_CUTOFF == pd.Timestamp("2026-06-11")
    )


def test_tournament_over_after_the_scheduled_end():
    wc = load_wc2026()
    assert not replay.tournament_over(wc, pd.Timestamp("2026-07-19"))  # final day
    assert replay.tournament_over(wc, pd.Timestamp("2026-07-20"))
    assert not replay.tournament_over({}, pd.Timestamp("2030-01-01"))


def test_pre_tournament_wc_strips_results_without_mutating():
    wc = load_wc2026()
    wc["known_results"] = [
        {"home": "Spain", "away": "Cape Verde", "home_score": 3, "away_score": 0}
    ]
    wc["known_ko_results"] = [{"home": "Spain", "away": "Argentina", "winner": "Spain"}]
    pre = replay.pre_tournament_wc(wc)
    assert pre["known_results"] == [] and pre["known_ko_results"] == []
    assert pre["groups"] == wc["groups"]
    assert len(wc["known_results"]) == 1 and len(wc["known_ko_results"]) == 1  # input untouched


def test_pre_tournament_matches_drops_cutoff_day_and_after(synthetic_matches):
    cutoff = synthetic_matches["date"].iloc[len(synthetic_matches) // 2]
    pre = replay.pre_tournament_matches(synthetic_matches, cutoff)
    assert len(pre) > 0
    assert pre["date"].max() < cutoff
    assert len(pre) == int((synthetic_matches["date"] < cutoff).sum())


def test_state_is_invariant_to_anything_after_the_cutoff(
    synthetic_matches, synthetic_rankings, synthetic_squads
):
    """Fabricated post-cutoff matches and a post-cutoff ranking edition must change nothing."""
    cutoff = synthetic_matches["date"].iloc[len(synthetic_matches) // 2]
    honest = synthetic_matches[synthetic_matches["date"] < cutoff]
    rk_honest = synthetic_rankings[synthetic_rankings["date"] <= cutoff]

    # The "future": the weakest-looking team thrashes everyone, plus a ranking edition crowning it.
    loser = honest["away_team"].iloc[0]
    future = honest.tail(40).copy()
    future["date"] = cutoff + pd.to_timedelta(range(len(future)), unit="D")
    future["home_team"], future["home_score"], future["away_score"] = loser, 9, 0
    future = future[future["away_team"] != loser]
    future["margin"], future["result"] = 9, "H"
    poisoned = pd.concat([synthetic_matches, future], ignore_index=True)
    rk_future = rk_honest.tail(1).assign(date=cutoff + pd.Timedelta(days=5), total_points=99_999.0)
    rk_poisoned = pd.concat([synthetic_rankings, rk_future], ignore_index=True)

    got = replay.build_pre_tournament_state(poisoned, rk_poisoned, synthetic_squads, cutoff)
    want = build_inference_state(honest.reset_index(drop=True), rk_honest, synthetic_squads)

    assert got.as_of < cutoff
    assert got.ratings == want.ratings
    assert got.team_form == want.team_form
    assert got.pair_hist == want.pair_hist
    assert got.team_last_date == want.team_last_date
    assert got.fifa_points == want.fifa_points
    assert 99_999.0 not in got.fifa_points.values()


def test_pre_tournament_rankings_drop_later_editions(synthetic_rankings):
    cutoff = synthetic_rankings["date"].iloc[len(synthetic_rankings) // 2]
    pre = replay.pre_tournament_rankings(synthetic_rankings, cutoff)
    assert pre["date"].max() <= cutoff
    assert replay.pre_tournament_rankings(None, cutoff) is None


def test_replay_simulation_is_blind_to_real_results():
    """Same seed → identical tables, whether or not the real results are attached to the input."""
    wc = load_wc2026()
    teams = _teams(wc)
    stub = StubPredictor(teams)

    with_results = dict(wc)
    a, b, c, d = wc["groups"]["A"]
    with_results["known_results"] = [{"home": a, "away": b, "home_score": 0, "away_score": 7}]
    with_results["known_ko_results"] = [{"home": c, "away": d, "winner": d}]

    blank = simulate.TournamentSimulator(stub, replay.pre_tournament_wc(wc)).run(300, seed=5)
    leaked = simulate.TournamentSimulator(stub, replay.pre_tournament_wc(with_results)).run(
        300, seed=5
    )
    pd.testing.assert_frame_equal(blank.table, leaked.table)
    assert blank.most_likely_final == leaked.most_likely_final


def test_committed_replay_model_predates_the_opener():
    if not config.REPLAY_MODEL_PATH.exists():
        pytest.skip("models/model_pre_wc.joblib not built (scripts/train_pre_wc.py)")
    artifact = model.load_model(config.REPLAY_MODEL_PATH)
    assert pd.Timestamp(artifact.trained_through) < replay.REPLAY_CUTOFF


# --------------------------------------------------------------------------------------
# Grading (post-run only)
# --------------------------------------------------------------------------------------
def _real_tournament(seed: int):
    """One simulated tournament as bare knockout results + the truth to recover from them."""
    wc = load_wc2026()
    teams = _teams(wc)
    sim = simulate.TournamentSimulator(StubPredictor(teams, seed=seed), wc, seed=seed)
    reached = sim.simulate_once()
    expected = dict.fromkeys(teams, "Group")
    for stage in simulate.STAGES:  # later stages overwrite earlier ones → furthest stage
        for t in reached[stage]:
            expected[t] = stage
    ko = []
    for i, (mid, (h, a)) in enumerate(sim._last_participants.items()):
        if i % 2:  # orientation in the feed is arbitrary
            h, a = a, h
        stage = sim.layout.match_row.get(mid, (None,))[0]  # the third-place match has no row
        ko.append(
            {
                "home": h,
                "away": a,
                "winner": sim._last_winner_team[mid],
                "round": FEED_ROUND.get(stage, "Match for third place"),
            }
        )
    return teams, ko, expected


# openfootball's round labels for the 2026 knockout rounds.
FEED_ROUND = {
    "R32": "Round of 32",
    "R16": "Round of 16",
    "QF": "Quarter-final",
    "SF": "Semi-final",
    "Final": "Final",
}


# Seed 1 is a regression case: a semi-final loser who also loses the third-place match, which a
# label-free "infer the round from the bracket" approach mistook for a finalist.
@pytest.mark.parametrize("seed", range(8))
def test_actual_stage_reached_recovers_the_bracket(seed):
    teams, ko, expected = _real_tournament(seed)
    assert len(ko) == 32  # 31 main-bracket ties + the third-place match
    assert replay.actual_stage_reached(ko, teams) == expected
    assert replay.actual_stage_reached(list(reversed(ko)), teams) == expected


def test_actual_stage_reached_needs_round_labels():
    teams, ko, _ = _real_tournament(0)
    assert replay.actual_stage_reached([{**r, "round": None} for r in ko], teams) is None


@pytest.mark.parametrize(
    ("label", "stage"),
    [
        ("Round of 32", "R32"),
        ("Round of 16", "R16"),
        ("Quarter-finals", "QF"),
        ("Semi-final", "SF"),
        ("Final", "Final"),
        ("Match for third place", None),
        ("Third-place play-off", None),
        (None, None),
    ],
)
def test_stage_of_round(label, stage):
    assert replay.stage_of_round(label) == stage


def test_actual_stage_reached_is_none_until_the_final_is_played():
    teams, ko, expected = _real_tournament(0)
    champion = next(t for t, s in expected.items() if s == "Champion")
    finalists = {t for t, s in expected.items() if s in ("Final", "Champion")}
    without_final = [r for r in ko if {r["home"], r["away"]} != finalists]
    assert replay.actual_stage_reached(without_final, teams) is None
    assert replay.actual_stage_reached([], teams) is None
    assert champion in finalists


def test_reality_check_scores_the_replay():
    table = pd.DataFrame(
        {
            "team": ["A", "B", "C", "D"],
            "group": ["X", "X", "Y", "Y"],
            "R32": [1.0, 1.0, 1.0, 0.5],
            "R16": [0.9, 0.8, 0.7, 0.2],
            "QF": [0.8, 0.7, 0.5, 0.1],
            "SF": [0.7, 0.6, 0.4, 0.05],
            "Final": [0.6, 0.5, 0.3, 0.02],
            "Champion": [0.5, 0.3, 0.19, 0.01],
        }
    )
    actual = {"A": "SF", "B": "Champion", "C": "Final", "D": "Group"}
    rc = replay.reality_check(table, actual)
    assert rc["champion"] == {"team": "B", "p": 0.3, "rank": 2}
    assert rc["favourite"] == {"team": "A", "p": 0.5}
    by_stage = {r["stage"]: r for r in rc["rounds"]}
    assert by_stage["Champion"] == {"stage": "Champion", "places": 1, "hits": 0}
    assert by_stage["Final"]["places"] == 2 and by_stage["Final"]["hits"] == 1  # top-2: A, B
    row = rc["teams"].set_index("team")
    assert row.loc["D", "p_actual"] == pytest.approx(0.5)  # P(out in groups) = 1 - P(R32)
    assert list(rc["teams"]["team"]) == ["B", "C", "A", "D"]  # furthest stage first
