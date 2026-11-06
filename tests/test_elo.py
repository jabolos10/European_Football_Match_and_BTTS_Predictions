"""
Tests for src/elo.py.

Run from the project root:  python -m pytest -q
"""
import pandas as pd
import pytest

from src.elo import EloParams, compute_elo, expected_home_score, goal_difference_multiplier

COLUMNS = ["date", "league", "season", "home_team", "away_team", "home_goals", "away_goals"]


def make_matches(rows):
    """Build a small match table from (date, league, season, home, away, home_goals, away_goals) rows."""
    return pd.DataFrame(rows, columns=COLUMNS)


def rating_of(final_ratings, team):
    return final_ratings.loc[final_ratings["team"] == team, "elo"].item()


def test_equal_teams_on_neutral_ground_are_a_coin_flip():
    assert expected_home_score(1500, 1500, home_advantage=0) == pytest.approx(0.5)


def test_home_advantage_makes_the_home_team_favourite():
    assert expected_home_score(1500, 1500, home_advantage=60) > 0.5


def test_goal_difference_multiplier():
    assert goal_difference_multiplier(1) == 1.0
    assert goal_difference_multiplier(-2) == 1.5
    assert goal_difference_multiplier(3) == pytest.approx(14 / 8)


def test_winner_takes_points_from_loser_and_the_total_stays_the_same():
    matches = make_matches([("2020-08-01", "E0", 2020, "A", "B", 2, 1)])
    _, final_ratings = compute_elo(matches)
    assert rating_of(final_ratings, "A") > 1500 > rating_of(final_ratings, "B")
    assert rating_of(final_ratings, "A") + rating_of(final_ratings, "B") == pytest.approx(3000)


def test_bigger_win_moves_more_points():
    _, narrow = compute_elo(make_matches([("2020-08-01", "E0", 2020, "A", "B", 1, 0)]))
    _, big = compute_elo(make_matches([("2020-08-01", "E0", 2020, "A", "B", 4, 0)]))
    assert rating_of(big, "A") > rating_of(narrow, "A")


def test_ratings_are_saved_before_the_match():
    """No peeking into the future: a match's own result must not be in its pre-match ratings."""
    matches = make_matches([
        ("2020-08-01", "E0", 2020, "A", "B", 3, 0),
        ("2020-08-08", "E0", 2020, "A", "C", 0, 0),
    ])
    with_elo, _ = compute_elo(matches)
    assert with_elo.loc[0, "elo_home"] == 1500  # first match: nobody has played yet
    assert with_elo.loc[1, "elo_home"] > 1500   # second match: A's rating already includes the 3-0


def test_promoted_team_starts_below_the_league_average():
    matches = make_matches([
        ("2020-08-01", "E0", 2020, "A", "B", 2, 0),
        ("2021-08-01", "E0", 2021, "Promoted", "A", 0, 0),
    ])
    with_elo, _ = compute_elo(matches, EloParams(new_team_gap=100))
    # A and B average 1500 (points only moved between them), so the new team starts at 1400
    assert with_elo.loc[1, "elo_home"] == pytest.approx(1400)


def test_season_regression_pulls_ratings_back_to_the_average():
    matches = make_matches([
        ("2020-08-01", "E0", 2020, "A", "B", 3, 0),
        ("2021-08-01", "E0", 2021, "A", "B", 0, 0),
    ])
    without, _ = compute_elo(matches, EloParams(season_regression=0.0))
    with_half, _ = compute_elo(matches, EloParams(season_regression=0.5))
    gain = without.loc[1, "elo_home"] - 1500
    assert with_half.loc[1, "elo_home"] - 1500 == pytest.approx(0.5 * gain)
