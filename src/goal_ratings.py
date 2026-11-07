"""
Dynamic attack and defence ratings - a "power ranking for goals".

Elo gives every team ONE number: how good it is overall. For "both teams to score" that
is not enough - we need to know separately how well a team ATTACKS and how well it DEFENDS.

Every team gets two numbers:
  attack  - how many goals it scores compared with an average team,
  defence - how few goals it lets in compared with an average team.

Expected goals of the home team = league average x home advantage x home attack / away defence
(written on a log scale, so "x" and "/" become "+" and "-").

After every match each number moves by learning_rate x (what happened - what we expected),
like Elo, but separately for attack and defence and adjusted for the opponent:
scoring 3 against a top defence raises the attack more than scoring 3 against the worst one.
This is the "score-driven" dynamic model of Koopman & Lit, i.e. a simple filter that tracks
a hidden state (team strength) from noisy measurements (goals) - like a Kalman filter.

The same code works for any count: goals, or shots on target (less luck than goals).
The key output: ratings of both teams BEFORE every match (no peeking into the future).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RatingParams:
    learning_rate: float = 0.05     # how fast attack/defence react to a surprise
    league_rate: float = 0.005      # how fast the league average and home advantage adapt
    season_regression: float = 0.3  # at a new season, move ratings this share back to average
    new_team_rating: float = -0.1   # promoted teams start slightly below average (attack and defence)


def compute_goal_ratings(matches, home_col="home_goals", away_col="away_goals", params=RatingParams(), prefix="g",
                         start_mean=1.35):
    """Go through all matches in date order and track attack/defence ratings for every team.

    Returns a copy of `matches` (same rows, same order) with these columns (all known BEFORE kick-off):
      {prefix}_att_home, {prefix}_def_home, {prefix}_att_away, {prefix}_def_away,
      {prefix}_exp_home, {prefix}_exp_away  - expected counts (e.g. expected goals) for both teams.
    Matches without a value in home_col/away_col (not played, or no statistics) get predictions
    but do not update the ratings.
    start_mean: a typical count per team per match to start every league with
    (about 1.35 for goals, about 4.3 for shots on target) - it then adapts to each league.
    """
    df = matches.copy()
    dates = pd.to_datetime(df["date"]).to_numpy()
    order = np.argsort(dates, kind="stable")  # go through the matches in date order
    rows = df[["league", "season", "home_team", "away_team", home_col, away_col]].to_numpy(object)

    attack, defence = {}, {}
    league_mean, league_home, league_season, league_first = {}, {}, {}, {}
    team_league = {}
    out = np.full((len(df), 6), np.nan)

    for i in order:
        league, season, home, away, home_count, away_count = rows[i]

        if league not in league_mean:
            league_mean[league] = np.log(start_mean)
            league_home[league] = 0.15
            league_first[league] = season
        if league in league_season and season != league_season[league]:
            for team, lg in team_league.items():  # new season: pull ratings back to average
                if lg == league:
                    attack[team] *= 1 - params.season_regression
                    defence[team] *= 1 - params.season_regression
        league_season[league] = season

        for team in (home, away):
            if team not in attack:
                start = 0.0 if season == league_first[league] else params.new_team_rating
                attack[team], defence[team] = start, start
            team_league[team] = league

        log_home = league_mean[league] + league_home[league] + attack[home] - defence[away]
        log_away = league_mean[league] + attack[away] - defence[home]
        expected_home, expected_away = np.exp(log_home), np.exp(log_away)
        out[i] = [attack[home], defence[home], attack[away], defence[away], expected_home, expected_away]

        if pd.isna(home_count) or pd.isna(away_count):
            continue
        surprise_home = home_count - expected_home  # > 0: home team scored more than expected
        surprise_away = away_count - expected_away
        rate = params.learning_rate
        attack[home] += rate * surprise_home
        defence[away] -= rate * surprise_home
        attack[away] += rate * surprise_away
        defence[home] -= rate * surprise_away
        league_mean[league] += params.league_rate * (surprise_home + surprise_away) / 2
        league_home[league] += params.league_rate * (surprise_home - surprise_away) / 2

    names = ["att_home", "def_home", "att_away", "def_away", "exp_home", "exp_away"]
    for j, name in enumerate(names):
        df[f"{prefix}_{name}"] = out[:, j]
    return df
