"""
Tests for src/goal_ratings.py and src/ratings_model.py.

Run from the project root:  python -m pytest -q
"""
import numpy as np
import pandas as pd

from src.goal_ratings import compute_goal_ratings
from src.goals_model import markets
from src.ratings_model import btts_dixon_coles


def matches(rows):
    return pd.DataFrame(rows, columns=["date", "season", "league", "home_team", "away_team", "home_goals", "away_goals"])


def test_ratings_use_only_earlier_matches_and_keep_row_order():
    df = matches([
        ["2024-08-20", 2024, "E0", "A", "B", 4, 0],   # listed first, played second
        ["2024-08-10", 2024, "E0", "A", "B", np.nan, np.nan],  # not played yet - no update
        ["2024-08-01", 2024, "E0", "A", "B", 3, 0],
    ])
    out = compute_goal_ratings(df)
    assert list(out["date"]) == list(df["date"])          # same order as the input
    assert out.loc[2, "g_att_home"] == 0                   # first match: nothing known yet
    assert out.loc[1, "g_att_home"] > 0                    # after A scored 3: attack went up
    assert out.loc[0, "g_att_home"] == out.loc[1, "g_att_home"]  # the unplayed match changed nothing


def test_dixon_coles_without_correction_equals_plain_poisson():
    home, away = np.array([0.8, 1.5, 2.4]), np.array([1.1, 0.9, 1.7])
    assert np.allclose(btts_dixon_coles(home, away, rho=0), markets(home, away)["btts_yes"], atol=1e-6)


def test_negative_rho_moves_chance_from_1_0_to_1_1():
    # rho < 0 makes 1-0 and 0-1 less likely and 0-0 and 1-1 more likely; 1-1 counts as BTTS, so BTTS goes up a bit
    home, away = np.array([1.4]), np.array([1.1])
    plain, corrected = btts_dixon_coles(home, away, rho=0), btts_dixon_coles(home, away, rho=-0.15)
    assert plain[0] < corrected[0] < plain[0] + 0.05


def test_result_chances_add_up_and_match_plain_poisson_without_correction():
    from src.ratings_model import result_dixon_coles
    home, away = np.array([0.8, 1.5, 2.4]), np.array([1.1, 0.9, 1.7])
    corrected = result_dixon_coles(home, away)
    assert np.allclose(corrected.sum(axis=1), 1)
    plain = result_dixon_coles(home, away, rho=0)
    goals = markets(home, away)
    assert np.allclose(plain, np.column_stack([goals["home_win"], goals["draw"], goals["away_win"]]), atol=1e-6)
    assert (corrected[:, 1] > plain[:, 1]).all()   # rho < 0: more draws (0-0 and 1-1), closer to real football
