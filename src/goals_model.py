"""
Goals model: predict how many goals each team will score, then turn that into probabilities.

Instead of asking "will both teams score?" directly, we predict two numbers:
  expected goals of the home team and expected goals of the away team
(Poisson regression: the standard way to model counts like goals).

From these two numbers we get the chance of every score (0-0, 1-0, 1-1, 2-1, ...) and from
that table everything at once: home win / draw / away win, BTTS, over 2.5 goals and
combinations like "home win and both teams score".
"""
import numpy as np
from scipy.stats import poisson
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MAX_GOALS = 10  # scores above 10 goals are so rare we can ignore them


def goals_regressor():
    """Fill missing values, put features on the same scale, Poisson regression."""
    return make_pipeline(SimpleImputer(), StandardScaler(), PoissonRegressor(alpha=1e-3, max_iter=1000))


def score_table(home_rate, away_rate):
    """Chance of every score for every match: shape (matches, home goals 0..10, away goals 0..10).

    The two teams' goals are treated as independent Poisson numbers.
    """
    goals = np.arange(MAX_GOALS + 1)
    home = poisson.pmf(goals, np.asarray(home_rate, dtype=float)[:, None])
    away = poisson.pmf(goals, np.asarray(away_rate, dtype=float)[:, None])
    table = home[:, :, None] * away[:, None, :]
    return table / table.sum(axis=(1, 2), keepdims=True)


def markets(home_rate, away_rate):
    """All the probabilities we need from the expected goals of both teams."""
    table = score_table(home_rate, away_rate)
    size = MAX_GOALS + 1
    home_wins = np.tril(np.ones((size, size)), k=-1).astype(bool)  # home goals > away goals
    away_wins = np.triu(np.ones((size, size)), k=1).astype(bool)   # away goals > home goals
    both_score = np.zeros((size, size), dtype=bool)
    both_score[1:, 1:] = True
    total = np.add.outer(np.arange(size), np.arange(size))
    return {
        "home_win": table[:, home_wins].sum(axis=1),
        "draw": np.trace(table, axis1=1, axis2=2),
        "away_win": table[:, away_wins].sum(axis=1),
        "btts_yes": table[:, both_score].sum(axis=1),
        "over_2_5": table[:, total > 2].sum(axis=1),
        "home_win_and_btts": table[:, home_wins & both_score].sum(axis=1),
        "away_win_and_btts": table[:, away_wins & both_score].sum(axis=1),
    }
