""" 
Scores for probability forecasts of match results

Every forecast is three probabilities in this order: home win, draw, away win.
The order matter for RPS beause the outcomes are ranked (a draw is "between" a home win and away win).
"""
import numpy as np

OUTCOMES = ["H", "D", "A"] #index 0 = home win, 1 = draw, 2 = away win

def outcome_index(home_goals, away_goals):
    #0 = home win, 1 = draw, 2 = away win
    home_goals = np.asarray(home_goals)
    away_goals = np.asarray(away_goals)
    return np.where(home_goals > away_goals, 0, np.where(home_goals == away_goals, 1, 2)) 

def one_hot(outcomes):
    #[0, 2] -> [[1, 0, 0], [0, 0, 1]]
    result = np.zeros((len(outcomes), 3))
    result[np.arange(len(outcomes)), outcomes] = 1.0
    return result

def rps(probs, outcomes):
    """
    Ranked probability score the standard score in football forecasting.

    Compares the cumulative forecast with the cumulative result.
    0 = perfect, lower is better.
    Unlike accuracy it rewards "almost right": giving a draw 40% when the home team won is punished less then giving the away team 40%.
    """
    probs = np.asarray(probs, dtype=float)
    observed  = one_hot(outcomes)
    cumulative_gap = np.cumsum(probs, axis=1) - np.cumsum(observed, axis=1)
    return float(np.mean(np.sum(cumulative_gap[:, :2] ** 2, axis=1) / 2))

def log_loss(probs, outcomes):
    """Average of -log(probability given to what actually happened). Lower is better."""
    probs = np.clip(np.asarray(probs, dtype=float), 1e-12, 1.0)
    return float(np.mean(-np.log(probs[np.arange(len(outcomes)), outcomes])))

def accuracy(probs, outcomes):
    """Share of matches where the most likely outcome happened."""
    return float(np.mean(np.argmax(probs, axis=1) == np.asarray(outcomes)))

def market_probabilities(odds_home, odds_draw, odds_away):
    """Turn bookmacher odds into probabilities.

    1 / odds give the bookmaker's probability plus its margin (the three add up to slightly more than 1), so we divide by the sum to remove the margin.
    """
    raw = 1.0 / np.column_stack([odds_home, odds_draw, odds_away]).astype(float)
    return raw / raw.sum(axis=1, keepdims=True)
