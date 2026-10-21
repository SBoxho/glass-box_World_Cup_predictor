"""Pre-tournament replay — re-simulate the finished 2026 World Cup with **no look-ahead**.

Once the tournament is over, the live simulator has nothing left to do: every match is locked to
its real result. The replay instead re-runs the whole tournament *as the model saw it on the eve
of the opener*, so its odds can be compared with what actually happened. That comparison is only
meaningful if no 2026 World Cup information leaks into the simulation, so every input is cut at
the opening day (:data:`REPLAY_CUTOFF`):

* **Model** — a separate artifact (``models/model_pre_wc.joblib``) trained on matches strictly
  before the cutoff (``scripts/train_pre_wc.py``); the production model was trained a few days into
  the group stage, so it is *not* reused here.
* **Team strength** (Elo, recent form, head-to-head, rest days) — replayed from matches strictly
  before the cutoff (:func:`pre_tournament_matches`).
* **FIFA ranking** (feature + group tiebreaker) — the latest snapshot dated on/before the cutoff
  (the pre-tournament edition); any later edition is dropped.
* **Tournament** — the December 2025 draw only: ``known_results`` / ``known_ko_results`` are
  emptied (:func:`pre_tournament_wc`), so no match is locked.
* **Squad ratings** — passed through: the committed EA FC 26 snapshot is a player-ratings database,
  not match results. It was captured on 17 Jun 2026, so the app flags it next to the replay.

The actual results are used for one thing only — grading the replay *after* it has run
(:func:`actual_stage_reached`, :func:`reality_check`). They never reach the simulator.

Framework-free like the rest of ``core``; the Streamlit layer adds caching and rendering.
"""

from __future__ import annotations

import pandas as pd

from core.features import InferenceState, build_inference_state
from core.simulate import STAGES

# Opening day of the 2026 World Cup (Mexico v South Africa, 11 Jun 2026). Matches are dated by day,
# so every match *on or after* this date is excluded; the ranking edition dated this day is the
# pre-tournament one (published before the opener) and is kept.
REPLAY_CUTOFF = pd.Timestamp("2026-06-11")

# Furthest-stage labels, worst → best ("Group" = eliminated in the group stage).
REACHED = ["Group", *STAGES]


def cutoff_from_wc(wc: dict) -> pd.Timestamp:
    """The replay cutoff: the tournament's start date from ``wc2026.json``, else :data:`REPLAY_CUTOFF`."""
    start = (wc.get("dates") or {}).get("start")
    return pd.Timestamp(start) if start else REPLAY_CUTOFF


def tournament_over(wc: dict, today: pd.Timestamp | None = None) -> bool:
    """True once ``today`` is past the tournament's scheduled end date (``wc["dates"]["end"]``)."""
    end = (wc.get("dates") or {}).get("end")
    if not end:
        return False
    today = pd.Timestamp.now(tz="UTC").tz_localize(None) if today is None else pd.Timestamp(today)
    return today.normalize() > pd.Timestamp(end)


# --------------------------------------------------------------------------------------
# Inputs, cut at the opener
# --------------------------------------------------------------------------------------
def pre_tournament_wc(wc: dict) -> dict:
    """A copy of the tournament definition with every locked group/knockout result removed."""
    out = dict(wc)
    out["known_results"] = []
    out["known_ko_results"] = []
    out["known_results_as_of"] = None
    return out


def pre_tournament_matches(
    matches: pd.DataFrame, cutoff: pd.Timestamp = REPLAY_CUTOFF
) -> pd.DataFrame:
    """Match history strictly before ``cutoff`` (drops every 2026 World Cup match)."""
    return matches[matches["date"] < pd.Timestamp(cutoff)].reset_index(drop=True)


def _on_or_before(df: pd.DataFrame | None, cutoff: pd.Timestamp) -> pd.DataFrame | None:
    if df is None or df.empty:
        return df
    return df[df["date"] <= pd.Timestamp(cutoff)].reset_index(drop=True)


def pre_tournament_rankings(
    rankings: pd.DataFrame | None, cutoff: pd.Timestamp = REPLAY_CUTOFF
) -> pd.DataFrame | None:
    """FIFA ranking snapshots dated on/before ``cutoff`` (drops any in- or post-tournament edition)."""
    return _on_or_before(rankings, cutoff)


def build_pre_tournament_state(
    matches: pd.DataFrame,
    rankings: pd.DataFrame | None = None,
    squads: pd.DataFrame | None = None,
    cutoff: pd.Timestamp = REPLAY_CUTOFF,
) -> InferenceState:
    """Team-strength state as of the eve of the opener (see the module docstring for each input).

    ``squads`` is passed through unchanged: the committed EA FC 26 snapshot is a player-ratings
    database rather than match results (it is dated by when it was captured, 17 Jun 2026) — the
    app discloses this next to the replay.
    """
    return build_inference_state(
        pre_tournament_matches(matches, cutoff),
        pre_tournament_rankings(rankings, cutoff),
        squads,
    )


# --------------------------------------------------------------------------------------
# Grading — only ever run *after* the simulation
# --------------------------------------------------------------------------------------
def stage_of_round(label: str | None) -> str | None:
    """Map a feed round label to a stage key (``"R32"`` … ``"Final"``); ``None`` for the third-place
    match or an unrecognised label. Matching is by keyword, so ``"Quarter-finals"`` / ``"Quarterfinal"``
    style variants all resolve."""
    s = (label or "").strip().lower()
    if not s or "third" in s or "3rd" in s:
        return None
    if "32" in s:
        return "R32"
    if "16" in s:
        return "R16"
    if "quarter" in s:
        return "QF"
    if "semi" in s:
        return "SF"
    if "final" in s:
        return "Final"
    return None


def actual_stage_reached(ko_results: list[dict], teams: list[str]) -> dict[str, str] | None:
    """Each team's furthest real stage (``REACHED`` label), or ``None`` until the final is played.

    Read straight off the feed's round labels (``round`` on each knockout result): a team reached
    every round it played in, and the winner of the final is the champion. The third-place match
    is ignored. Returns ``None`` unless the final is on file and exactly 32 teams played the knockout
    rounds — i.e. only once the real tournament is complete.
    """
    level = {s: i for i, s in enumerate(REACHED)}
    out = dict.fromkeys(teams, "Group")
    final = None
    for r in ko_results:
        stage = stage_of_round(r.get("round"))
        if stage is None:
            continue
        for t in (r.get("home"), r.get("away")):
            if t in out and level[stage] > level[out[t]]:
                out[t] = stage
        if stage == "Final":
            final = r

    if final is None or final.get("winner") not in (final.get("home"), final.get("away")):
        return None
    if final["winner"] not in out or sum(s != "Group" for s in out.values()) != 32:
        return None
    out[final["winner"]] = "Champion"
    return out


def reality_check(table: pd.DataFrame, actual: dict[str, str]) -> dict:
    """Grade a replay's stage table against what actually happened.

    ``table`` is :attr:`core.simulate.SimulationResult.table` (one row per team with the probability
    of reaching each stage); ``actual`` comes from :func:`actual_stage_reached`. Returns:

    * ``champion`` — the real champion, the replay's title odds for them and their odds rank;
    * ``favourite`` — the replay's own title favourite and its odds;
    * ``rounds`` — per stage (R16 → Final), how many of the teams that really got there were among
      the replay's top-k by probability of reaching it (k = the number of places in that round);
    * ``teams`` — one row per team: furthest real stage + the replay's probability of reaching it.
    """
    t = table.sort_values("Champion", ascending=False).reset_index(drop=True)
    title_rank = {team: i + 1 for i, team in enumerate(t["team"])}
    p_title = dict(zip(t["team"], t["Champion"], strict=True))

    champ = next((team for team, s in actual.items() if s == "Champion"), None)
    fav = t.iloc[0]

    rounds = []
    for stage in ["R16", "QF", "SF", "Final", "Champion"]:
        level = REACHED.index(stage)
        really = {team for team, s in actual.items() if REACHED.index(s) >= level}
        k = len(really)
        top_k = set(t.sort_values(stage, ascending=False)["team"].head(k))
        rounds.append({"stage": stage, "places": k, "hits": len(really & top_k)})

    rows = []
    for _, r in t.iterrows():
        s = actual.get(r["team"], "Group")
        p_reach = 1.0 - float(r["R32"]) if s == "Group" else float(r[s])
        rows.append(
            {
                "team": r["team"],
                "group": r["group"],
                "actual": s,
                "p_actual": p_reach,
                "p_title": float(r["Champion"]),
                "title_rank": title_rank[r["team"]],
            }
        )
    teams = pd.DataFrame(rows)
    teams["_lvl"] = teams["actual"].map(REACHED.index)
    teams = (
        teams.sort_values(["_lvl", "p_title"], ascending=[False, False])
        .drop(columns="_lvl")
        .reset_index(drop=True)
    )

    return {
        "champion": None
        if champ is None
        else {"team": champ, "p": float(p_title.get(champ, 0.0)), "rank": title_rank.get(champ)},
        "favourite": {"team": fav["team"], "p": float(fav["Champion"])},
        "rounds": rounds,
        "teams": teams,
    }
