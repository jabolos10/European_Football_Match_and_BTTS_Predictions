"""Elo ratings for football clubs, computed only from match results.

How it works:
  - every team has one number, its rating (a typical team is around 1500),
  - before a match we compute how likely the home team is to win from the rating
    difference plus a home advantage bonus,
  - after the match the winner takes points from the loser; a surprising result
    (an underdog win) moves more points than an expected one,
  - bigger wins move a bit more points than narrow ones.

The key output is the PRE-match rating of both teams for every match:
a model may only use information that was known before kick-off.
"""
from dataclasses import dataclass

import pandas as pd

REQUIRED_COLUMNS = ["date", "league", "season", "home_team", "away_team", "home_goals", "away_goals"]


@dataclass(frozen=True)
class EloParams:
    k: float = 20.0                 # how many rating points can move in one match
    home_advantage: float = 60.0    # bonus (in rating points) the home team gets when we compute chances
    initial_rating: float = 1500.0  # every team in a league's first season in our data starts here
    new_team_gap: float = 100.0     # a newly promoted team starts this far below its league's average
    season_regression: float = 0.0  # e.g. 0.2 = at a new season, move ratings 20% back to the league average


def expected_home_score(elo_home, elo_away, home_advantage):
    """Expected result for the home team: from 0 (sure loss) through 0.5 to 1 (sure win)."""
    return 1.0 / (1.0 + 10 ** ((elo_away - elo_home - home_advantage) / 400.0))


def goal_difference_multiplier(goal_difference):
    """Bigger wins move more points (the rule used by the World Football Elo Ratings)."""
    margin = abs(goal_difference)
    if margin <= 1:
        return 1.0
    if margin == 2:
        return 1.5
    return (11.0 + margin) / 8.0


def match_result(home_goals, away_goals):
    """Result from the home team's point of view: 1 = win, 0.5 = draw, 0 = loss."""
    if home_goals > away_goals:
        return 1.0
    if home_goals == away_goals:
        return 0.5
    return 0.0


def compute_elo(matches, params=EloParams()):
    """Go through all matches in date order and compute Elo ratings.

    Returns:
      matches_with_elo - the input table sorted by date, plus elo_home and elo_away:
                         the ratings of both teams BEFORE each match,
      final_ratings    - one row per team with its rating after its last match.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in matches.columns]
    if missing:
        raise ValueError(f"matches is missing columns: {missing}")

    df = matches.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date", kind="stable").reset_index(drop=True)

    ratings = {}              # team -> current rating
    team_league = {}          # team -> league it played in most recently
    league_season = {}        # league -> season of its latest match
    league_first_season = {}  # league -> its first season in our data

    elo_home, elo_away = [], []
    for row in df.itertuples(index=False):
        league, season = row.league, row.season

        # 1) A new season starts in this league: optionally pull ratings back to the average
        if league in league_season and season != league_season[league]:
            _regress_to_league_mean(ratings, team_league, league, params.season_regression)
        league_season[league] = season
        league_first_season.setdefault(league, season)

        # 2) A team we have never seen gets a starting rating
        for team in (row.home_team, row.away_team):
            if team not in ratings:
                if season == league_first_season[league]:
                    ratings[team] = params.initial_rating
                else:  # it came up from a lower division we don't have data for
                    average = _league_mean(ratings, team_league, league, params.initial_rating)
                    ratings[team] = average - params.new_team_gap
            team_league[team] = league

        # 3) Save the ratings BEFORE the match - this is what a model is allowed to see
        home_rating, away_rating = ratings[row.home_team], ratings[row.away_team]
        elo_home.append(home_rating)
        elo_away.append(away_rating)

        # 4) After the match the winner takes points from the loser
        expected = expected_home_score(home_rating, away_rating, params.home_advantage)
        actual = match_result(row.home_goals, row.away_goals)
        multiplier = goal_difference_multiplier(row.home_goals - row.away_goals)
        change = params.k * multiplier * (actual - expected)
        ratings[row.home_team] = home_rating + change
        ratings[row.away_team] = away_rating - change

    df["elo_home"] = elo_home
    df["elo_away"] = elo_away

    final_ratings = pd.DataFrame({
        "team": list(ratings),
        "league": [team_league[team] for team in ratings],
        "elo": list(ratings.values()),
    }).sort_values("elo", ascending=False, ignore_index=True)
    return df, final_ratings


def _league_mean(ratings, team_league, league, default):
    """Average rating of the teams that most recently played in this league."""
    values = [ratings[team] for team, lg in team_league.items() if lg == league]
    return sum(values) / len(values) if values else default


def _regress_to_league_mean(ratings, team_league, league, share):
    """Move every rating in the league `share` of the way back to the league average."""
    if share <= 0:
        return
    average = _league_mean(ratings, team_league, league, default=0.0)
    for team, lg in team_league.items():
        if lg == league:
            ratings[team] = average + (1.0 - share) * (ratings[team] - average)
