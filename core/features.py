"""Point-in-time feature engineering — no leakage by construction.

Every feature for a match dated ``D`` is computed from matches strictly before ``D``. The
same low-level aggregation helpers back both code paths:

* :func:`build_features` — one efficient chronological pass over the whole history, used to
  build the training table.
* :func:`features_for_match` — recompute a single match's features from an arbitrary history
  slice, used at inference time and by the no-leakage guardrail test.

Because both paths call the identical helpers (and the identical Elo replay), recomputing a
sampled match from a truncated history must reproduce the pipeline's row exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core import config
from core.elo import compute_elo, ratings_as_of

# Identifier columns carried alongside features for training / filtering / display.
# (``neutral`` is intentionally not repeated here — it is already a model feature.)
META_COLUMNS = ["date", "year", "home_team", "away_team", "result", "tournament"]


# --------------------------------------------------------------------------------------
# Low-level aggregations (shared by batch + single-match paths — keep them pure)
# --------------------------------------------------------------------------------------
def _ppg_gd(form_list: list[tuple[float, float]]) -> tuple[float, float]:
    """Points-per-game and average goal difference over the last FORM_WINDOW entries.

    ``form_list`` holds ``(points, goal_diff)`` from a team's matches in chronological order.
    Defaults (no history): a neutral 1.0 ppg and 0.0 goal difference.
    """
    if not form_list:
        return 1.0, 0.0
    window = form_list[-config.FORM_WINDOW :]
    pts = sum(p for p, _ in window) / len(window)
    gd = sum(g for _, g in window) / len(window)
    return pts, gd


def _h2h_winrate(entries: list[tuple[pd.Timestamp, str | None]], home: str, ref_date) -> float:
    """Decayed head-to-head score-share from the home team's perspective.

    ``entries`` are ``(date, winner)`` for past meetings of the two sides (winner is a team
    name, or ``None`` for a draw). Recent meetings count more (exponential decay with
    ``H2H_HALFLIFE_YEARS``). Returns ``H2H_PRIOR`` when the sides have never met.
    """
    num = 0.0
    den = 0.0
    for date, winner in entries:
        years = (ref_date - date).days / 365.25
        w = 0.5 ** (years / config.H2H_HALFLIFE_YEARS)
        share = 0.5 if winner is None else (1.0 if winner == home else 0.0)
        num += w * share
        den += w
    return num / den if den > 0 else config.H2H_PRIOR


def _rest_days(last_date, ref_date) -> float:
    """Days since a team's previous match, capped at ``MAX_REST_DAYS``."""
    if last_date is None:
        return float(config.MAX_REST_DAYS)
    return float(min((ref_date - last_date).days, config.MAX_REST_DAYS))


def _outcome_points(team_is_home: bool, result: str) -> int:
    """3/1/0 points for a team given the {H,D,A} result and whether it was the home side."""
    if result == "D":
        return 1
    home_won = result == "H"
    return 3 if (home_won == team_is_home) else 0


# --------------------------------------------------------------------------------------
# Batch pipeline
# --------------------------------------------------------------------------------------
def build_features(matches: pd.DataFrame, train_start_year: int | None = None) -> pd.DataFrame:
    """Build the full feature table from a cleaned, date-sorted match frame.

    Elo is added in a batched pass; the rolling form / head-to-head / rest features are then
    accumulated in a single date-batched loop (state is updated only *after* a day's features
    are read, so same-day matches never see each other). Optionally filters the returned rows
    to ``year >= train_start_year`` while still having used full history to warm up the state.
    """
    df = compute_elo(matches)

    team_form: dict[str, list[tuple[float, float]]] = {}
    team_last_date: dict[str, pd.Timestamp] = {}
    pair_hist: dict[frozenset, list[tuple[pd.Timestamp, str | None]]] = {}

    n = len(df)
    form_home = np.empty(n)
    form_away = np.empty(n)
    gd_home = np.empty(n)
    gd_away = np.empty(n)
    h2h = np.empty(n)
    rest_home = np.empty(n)
    rest_away = np.empty(n)

    positions = df.index.to_list()
    pos_of = {idx: i for i, idx in enumerate(positions)}

    for date, day in df.groupby("date", sort=True):
        # 1) Read features for every match on this day from pre-day state.
        for idx, row in day.iterrows():
            i = pos_of[idx]
            h, a = row["home_team"], row["away_team"]
            fh, gh = _ppg_gd(team_form.get(h, []))
            fa, ga = _ppg_gd(team_form.get(a, []))
            form_home[i], gd_home[i] = fh, gh
            form_away[i], gd_away[i] = fa, ga
            h2h[i] = _h2h_winrate(pair_hist.get(frozenset((h, a)), []), h, date)
            rest_home[i] = _rest_days(team_last_date.get(h), date)
            rest_away[i] = _rest_days(team_last_date.get(a), date)

        # 2) Now fold this day's results into the running state.
        for _, row in day.iterrows():
            h, a, res = row["home_team"], row["away_team"], row["result"]
            gd = int(row["margin"])
            team_form.setdefault(h, []).append((_outcome_points(True, res), gd))
            team_form.setdefault(a, []).append((_outcome_points(False, res), -gd))
            team_last_date[h] = date
            team_last_date[a] = date
            winner = None if res == "D" else (h if res == "H" else a)
            pair_hist.setdefault(frozenset((h, a)), []).append((date, winner))

    out = df.copy()
    out["form_home"] = form_home
    out["form_away"] = form_away
    out["gd_home"] = gd_home
    out["gd_away"] = gd_away
    out["h2h_home_winrate"] = h2h
    out["days_rest_home"] = rest_home
    out["days_rest_away"] = rest_away
    # is_host_home before neutral is coerced to int (reads the boolean venue flag + tournament).
    out["is_host_home"] = [
        int((not bool(nt)) and config.is_finals_tournament(tn))
        for nt, tn in zip(out["neutral"], out["tournament"], strict=False)
    ]
    out["neutral"] = out["neutral"].astype(int)

    cols = config.FEATURES + [c for c in META_COLUMNS if c not in config.FEATURES]
    result = out[cols]
    if train_start_year is not None:
        result = result[result["year"] >= train_start_year].reset_index(drop=True)
    return result.reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Single-match recompute (inference + leakage test)
# --------------------------------------------------------------------------------------
def features_for_match(
    history: pd.DataFrame,
    home: str,
    away: str,
    date: pd.Timestamp,
    neutral: bool,
    is_host_home: int | bool,
) -> dict[str, float]:
    """Compute one match's features from a history frame, using only rows before ``date``.

    ``history`` may contain anything; only matches strictly before ``date`` are used. Returns
    a dict keyed by :data:`core.config.FEATURES`. This is the inference path and the reference
    implementation the no-leakage test checks ``build_features`` against.
    """
    date = pd.Timestamp(date)
    hist = history[history["date"] < date]

    ratings = ratings_as_of(hist, None)  # hist already excludes >= date
    elo_diff = ratings.get(home, config.ELO_BASE) - ratings.get(away, config.ELO_BASE)

    form_home, gd_home = _ppg_gd(_team_form_list(hist, home))
    form_away, gd_away = _ppg_gd(_team_form_list(hist, away))

    pair = hist[
        ((hist["home_team"] == home) & (hist["away_team"] == away))
        | ((hist["home_team"] == away) & (hist["away_team"] == home))
    ]
    entries = [
        (
            r["date"],
            None
            if r["result"] == "D"
            else (r["home_team"] if r["result"] == "H" else r["away_team"]),
        )
        for _, r in pair.iterrows()
    ]
    h2h = _h2h_winrate(entries, home, date)

    rest_home = _rest_days(_last_date(hist, home), date)
    rest_away = _rest_days(_last_date(hist, away), date)

    return {
        "elo_diff": elo_diff,
        "form_home": form_home,
        "form_away": form_away,
        "gd_home": gd_home,
        "gd_away": gd_away,
        "h2h_home_winrate": h2h,
        "neutral": int(bool(neutral)),
        "is_host_home": int(bool(is_host_home)),
        "days_rest_home": rest_home,
        "days_rest_away": rest_away,
    }


def _team_form_list(hist: pd.DataFrame, team: str) -> list[tuple[float, float]]:
    """Rebuild a team's chronological ``(points, goal_diff)`` list from a history slice."""
    rows = hist[(hist["home_team"] == team) | (hist["away_team"] == team)]
    out: list[tuple[float, float]] = []
    for _, r in rows.iterrows():
        is_home = r["home_team"] == team
        gd = int(r["margin"]) if is_home else -int(r["margin"])
        out.append((_outcome_points(is_home, r["result"]), gd))
    return out


def _last_date(hist: pd.DataFrame, team: str):
    rows = hist[(hist["home_team"] == team) | (hist["away_team"] == team)]
    if rows.empty:
        return None
    return rows["date"].max()


# --------------------------------------------------------------------------------------
# Inference state — current team strength precomputed once for fast, repeated predictions
# --------------------------------------------------------------------------------------
@dataclass
class InferenceState:
    """Snapshot of current team strength built from the full match history in one pass.

    The Streamlit app builds this once (cached) and asks it for many feature rows — the
    tournament simulator alone needs a prediction for every plausible matchup. Reusing the
    same aggregation helpers as :func:`build_features` keeps inference consistent with training.
    """

    ratings: dict[str, float]
    team_form: dict[str, list]
    team_last_date: dict[str, pd.Timestamp]
    pair_hist: dict[frozenset, list]
    as_of: pd.Timestamp
    teams: list[str] = field(default_factory=list)

    def feature_row(
        self,
        home: str,
        away: str,
        neutral: bool,
        is_host_home: int | bool,
        ref_date: pd.Timestamp | None = None,
    ) -> dict[str, float]:
        """Build a single feature dict for a (future) match from current state."""
        ref = pd.Timestamp(ref_date) if ref_date is not None else self.as_of + pd.Timedelta(days=1)
        fh, gh = _ppg_gd(self.team_form.get(home, []))
        fa, ga = _ppg_gd(self.team_form.get(away, []))
        return {
            "elo_diff": self.ratings.get(home, config.ELO_BASE)
            - self.ratings.get(away, config.ELO_BASE),
            "form_home": fh,
            "form_away": fa,
            "gd_home": gh,
            "gd_away": ga,
            "h2h_home_winrate": _h2h_winrate(
                self.pair_hist.get(frozenset((home, away)), []), home, ref
            ),
            "neutral": int(bool(neutral)),
            "is_host_home": int(bool(is_host_home)),
            "days_rest_home": _rest_days(self.team_last_date.get(home), ref),
            "days_rest_away": _rest_days(self.team_last_date.get(away), ref),
        }


def build_inference_state(matches: pd.DataFrame) -> InferenceState:
    """Replay the full history once to capture each team's current strength state."""
    ratings = ratings_as_of(matches, None)
    team_form: dict[str, list] = {}
    team_last_date: dict[str, pd.Timestamp] = {}
    pair_hist: dict[frozenset, list] = {}

    for _, row in matches.iterrows():
        h, a, res = row["home_team"], row["away_team"], row["result"]
        gd = int(row["margin"])
        team_form.setdefault(h, []).append((_outcome_points(True, res), gd))
        team_form.setdefault(a, []).append((_outcome_points(False, res), -gd))
        team_last_date[h] = row["date"]
        team_last_date[a] = row["date"]
        winner = None if res == "D" else (h if res == "H" else a)
        pair_hist.setdefault(frozenset((h, a)), []).append((row["date"], winner))

    return InferenceState(
        ratings=ratings,
        team_form=team_form,
        team_last_date=team_last_date,
        pair_hist=pair_hist,
        as_of=matches["date"].max(),
        teams=sorted(set(matches["home_team"]) | set(matches["away_team"])),
    )
