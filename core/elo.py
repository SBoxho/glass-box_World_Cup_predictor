"""Rolling Elo computed from full match history.

Chess-style update with two football-specific touches (eloratings.net-inspired):
the K-factor scales with tournament importance and goal margin, and the home side gets a
rating bonus in the *expectation* that is zeroed at neutral venues. Stored ratings are
therefore venue-neutral skill estimates; the model learns venue effects separately.

Leakage discipline: ratings are updated **date-batch by date-batch**, so the pre-match
rating used for a match on date D reflects only matches strictly before D — never another
match played the same day. This is what makes ``elo_diff`` pass the no-leakage test.
"""

from __future__ import annotations

import pandas as pd

from core.config import (
    ELO_BASE,
    ELO_HOME_ADV,
    goal_difference_multiplier,
    tournament_k,
)


class EloEngine:
    """Mutable Elo state. Replay matches in chronological order to evolve ratings."""

    def __init__(self, base: float = ELO_BASE, home_adv: float = ELO_HOME_ADV):
        self.base = base
        self.home_adv = home_adv
        self.ratings: dict[str, float] = {}

    def rating(self, team: str) -> float:
        return self.ratings.get(team, self.base)

    def expected_home(self, home: str, away: str, neutral: bool) -> float:
        """Expected score for the home side in [0, 1] (win=1, draw=0.5, loss=0)."""
        ha = 0.0 if neutral else self.home_adv
        return 1.0 / (1.0 + 10.0 ** ((self.rating(away) - (self.rating(home) + ha)) / 400.0))

    def update(
        self,
        home: str,
        away: str,
        home_score: int,
        away_score: int,
        neutral: bool,
        tournament: str,
    ) -> None:
        """Apply one match's zero-sum Elo update."""
        exp_home = self.expected_home(home, away, neutral)
        margin = int(home_score) - int(away_score)
        actual = 1.0 if margin > 0 else (0.5 if margin == 0 else 0.0)
        k = tournament_k(tournament) * goal_difference_multiplier(margin)
        delta = k * (actual - exp_home)
        self.ratings[home] = self.rating(home) + delta
        self.ratings[away] = self.rating(away) - delta


def compute_elo(matches: pd.DataFrame) -> pd.DataFrame:
    """Return ``matches`` with pre-match Elo columns appended.

    Adds ``home_elo_pre``, ``away_elo_pre`` and ``elo_diff`` (home minus away), each computed
    from matches strictly before the row's date. ``matches`` must be sorted ascending by date.
    """
    engine = EloEngine()
    home_pre = [0.0] * len(matches)
    away_pre = [0.0] * len(matches)

    # Group by date so all matches on a day are scored against the same pre-day ratings,
    # then updated together — no same-day leakage.
    positions = matches.index.to_list()
    pos_of = {idx: i for i, idx in enumerate(positions)}

    for _, day in matches.groupby("date", sort=True):
        for idx, row in day.iterrows():
            i = pos_of[idx]
            home_pre[i] = engine.rating(row["home_team"])
            away_pre[i] = engine.rating(row["away_team"])
        for _, row in day.iterrows():
            engine.update(
                row["home_team"],
                row["away_team"],
                row["home_score"],
                row["away_score"],
                bool(row["neutral"]),
                row["tournament"],
            )

    out = matches.copy()
    out["home_elo_pre"] = home_pre
    out["away_elo_pre"] = away_pre
    out["elo_diff"] = out["home_elo_pre"] - out["away_elo_pre"]
    return out


def ratings_as_of(
    matches: pd.DataFrame, as_of_date: pd.Timestamp | None = None
) -> dict[str, float]:
    """Replay history and return the rating of every team as of a date.

    Includes only matches with ``date < as_of_date`` (strictly before). With ``as_of_date=None``
    the full history is replayed, giving each team's current/latest rating.
    """
    engine = EloEngine()
    df = matches if as_of_date is None else matches[matches["date"] < as_of_date]
    for _, row in df.iterrows():
        engine.update(
            row["home_team"],
            row["away_team"],
            row["home_score"],
            row["away_score"],
            bool(row["neutral"]),
            row["tournament"],
        )
    return dict(engine.ratings)


def elo_as_of(matches: pd.DataFrame, team: str, as_of_date: pd.Timestamp | None = None) -> float:
    """Elo rating of a single ``team`` as of a date (see :func:`ratings_as_of`)."""
    return ratings_as_of(matches, as_of_date).get(team, ELO_BASE)
