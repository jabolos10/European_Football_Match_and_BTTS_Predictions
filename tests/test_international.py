"""
Tests for src/international.py and src/derbies.py.

Run from the project root:  python -m pytest -q
"""
import pandas as pd

from src.derbies import is_derby
from src.international import compute_elo, goal_factor, k_factor


def test_derby_works_in_both_directions():
    assert is_derby("Arsenal", "Tottenham") and is_derby("Tottenham", "Arsenal")
    assert is_derby("Wisla", "Cracovia")
    assert not is_derby("Arsenal", "Legia")


def test_bigger_tournaments_and_bigger_wins_move_ratings_more():
    assert k_factor("FIFA World Cup") > k_factor("UEFA Nations League") > k_factor("Friendly")
    assert k_factor("UEFA Euro qualification") == k_factor("UEFA Nations League")
    assert goal_factor(1) < goal_factor(2) < goal_factor(4)


def test_elo_uses_only_earlier_matches_and_home_advantage():
    matches = pd.DataFrame({
        "date": pd.to_datetime(["2024-09-01", "2024-10-01"]),
        "home_team": ["Poland", "Poland"], "away_team": ["Sweden", "Sweden"],
        "home_goals": [2, 0], "away_goals": [0, 0],
        "tournament": ["UEFA Nations League", "Friendly"], "neutral": [False, True],
    })
    out, ratings = compute_elo(matches)
    assert out.loc[0, "elo_home"] == out.loc[0, "elo_away"] == 1500      # nothing known before the first match
    assert out.loc[1, "elo_home"] > out.loc[1, "elo_away"]              # Poland won the first one
    # second match: a draw on neutral ground between a stronger and a weaker team -> Poland loses a bit
    assert ratings["Poland"] < out.loc[1, "elo_home"]
