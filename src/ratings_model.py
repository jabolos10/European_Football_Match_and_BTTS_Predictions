"""
The "ratings" model - a second model that runs next to the current ("classic") one.

Current goals model: Poisson regression on all match features (form, shots, venue, ...).
Ratings model:      Poisson regression on attack/defence ratings that are updated after every match
                    (src/goal_ratings.py), for goals AND for shots on target, plus Elo and the league.
                    The score table then gets the Dixon-Coles correction: plain Poisson treats both
                    teams' goals as independent and so gets low scores (0-0, 1-0, 0-1, 1-1) slightly
                    wrong - exactly the scores that decide "both teams to score".

Walk-forward test (top 5 leagues + Ekstraklasa, 2021-22 to 2025-26, about 11,000 matches):
  BTTS log loss: ratings model 0.6854, current goals model 0.6874 (lower is better)
  10% lowest forecasts - both teams scored in: ratings model 47.2%, goals model 50.2%
The ratings model is better overall and at spotting "no BTTS" matches, but not in every league
(slightly worse in Ekstraklasa, where there are no shot statistics).

Match result (1X2): the ratings model alone is about as good as the classic logistic regression,
but the AVERAGE of both is better in every one of the 6 leagues (each model makes slightly
different mistakes, so averaging cancels some of them out):
  RPS classic 0.2011, ratings 0.2010, average of both 0.2006 (lower is better)
So the "Ratings" variant on the website = BTTS from this model + 1X2 from the average.
Both variants run side by side on the website until new matches show which one to keep.
"""
import numpy as np
from scipy.stats import poisson

from src.goal_ratings import RatingParams, compute_goal_ratings
from src.goals_model import goals_regressor

# Chosen on the validation seasons 2019-20 and 2020-21 (before the test seasons)
GOAL_PARAMS = RatingParams(learning_rate=0.02, season_regression=0.4)
SHOT_PARAMS = RatingParams(learning_rate=0.01, season_regression=0.3)
SHOTS_START_MEAN = 4.3  # typical shots on target per team per match
DIXON_COLES_RHO = -0.15  # < 0: 0-0 and 1-1 a bit more likely, 1-0 and 0-1 a bit less than plain Poisson (real football)
MAX_GOALS = 10

RATING_COLUMNS = [f"{kind}_{name}" for kind in ("g", "s")
                  for name in ("att_home", "def_home", "att_away", "def_away", "exp_home", "exp_away")]


def add_ratings(matches):
    """Add goal ratings (g_...) and shots-on-target ratings (s_...) to every match, known before kick-off."""
    df = compute_goal_ratings(matches, "home_goals", "away_goals", GOAL_PARAMS, prefix="g")
    return compute_goal_ratings(df, "home_shots_target", "away_shots_target", SHOT_PARAMS, prefix="s",
                                start_mean=SHOTS_START_MEAN)


def model_columns(leagues):
    return ["elo_diff"] + [f"league_{code}" for code in leagues] + RATING_COLUMNS


def score_table(home_rate, away_rate, rho=DIXON_COLES_RHO):
    """Chance of every exact score (home goals x away goals), with the Dixon-Coles low-score correction."""
    home_rate, away_rate = np.asarray(home_rate, float), np.asarray(away_rate, float)
    goals = np.arange(MAX_GOALS + 1)
    table = poisson.pmf(goals, home_rate[:, None])[:, :, None] * poisson.pmf(goals, away_rate[:, None])[:, None, :]
    table[:, 0, 0] *= 1 - home_rate * away_rate * rho
    table[:, 0, 1] *= 1 + home_rate * rho
    table[:, 1, 0] *= 1 + away_rate * rho
    table[:, 1, 1] *= 1 - rho
    table = np.clip(table, 0, None)
    return table / table.sum(axis=(1, 2), keepdims=True)


def btts_dixon_coles(home_rate, away_rate, rho=DIXON_COLES_RHO):
    """Chance that both teams score: every score where both have at least one goal."""
    return score_table(home_rate, away_rate, rho)[:, 1:, 1:].sum(axis=(1, 2))


def result_dixon_coles(home_rate, away_rate, rho=DIXON_COLES_RHO):
    """Home win / draw / away win chances (one row per match) from the score table."""
    table = score_table(home_rate, away_rate, rho)
    return np.column_stack([np.tril(table, -1).sum(axis=(1, 2)),        # home scored more
                            np.trace(table, axis1=1, axis2=2),          # same number of goals
                            np.triu(table, 1).sum(axis=(1, 2))])        # away scored more


def fit_predict(train, upcoming, leagues):
    """Train on finished matches and predict upcoming ones.

    Returns (BTTS chance, home/draw/away chances) of the ratings model for every upcoming match.
    """
    columns = [c for c in model_columns(leagues) if train[c].notna().any()]
    home_rate = goals_regressor().fit(train[columns], train["home_goals"]).predict(upcoming[columns])
    away_rate = goals_regressor().fit(train[columns], train["away_goals"]).predict(upcoming[columns])
    return btts_dixon_coles(home_rate, away_rate), result_dixon_coles(home_rate, away_rate)
