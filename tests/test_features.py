"""
Tests for src/features.py - mainly: no feature may use the match it describes or later ones.

Run from the project root:  python -m pytest -q
"""
import pandas as pd
import pytest

from src.features import add_features


def make_matches(rows):
    """(date, league, home, away, home_goals, away_goals) rows -> table with Elo columns filled in."""
    df = pd.DataFrame(rows, columns=["date", "league", "home_team", "away_team", "home_goals", "away_goals"])
    df["elo_home"], df["elo_away"] = 1500.0, 1500.0
    return df


MATCHES = make_matches([
    ("2024-08-03", "E0", "A", "B", 2, 1),   # Saturday
    ("2024-08-10", "E0", "B", "A", 0, 0),
    ("2024-08-17", "E0", "A", "C", 3, 3),
    ("2024-08-24", "E0", "A", "B", 1, 0),
])


def test_first_match_has_no_history():
    df = add_features(MATCHES, ["E0"])
    assert pd.isna(df.loc[0, "home_ppg"])
    assert pd.isna(df.loc[0, "h2h_home_ppg"])
    assert df.loc[0, "h2h_count"] == 0


def test_form_uses_only_earlier_matches():
    df = add_features(MATCHES, ["E0"])
    # Before match 3, A played: 2-1 win (3 pts), 0-0 draw (1 pt) -> 2 points per game
    assert df.loc[2, "home_ppg"] == pytest.approx(2.0)
    assert df.loc[2, "home_goals_for"] == pytest.approx(1.0)  # (2 + 0) / 2


def test_head_to_head_from_the_home_teams_view():
    df = add_features(MATCHES, ["E0"])
    # Before the last match A vs B met twice: A won 2-1 (3 pts) and drew 0-0 (1 pt)
    assert df.loc[3, "h2h_home_ppg"] == pytest.approx(2.0)
    assert df.loc[3, "h2h_btts_rate"] == pytest.approx(0.5)
    assert df.loc[3, "h2h_count"] == 2


def test_league_trend_ignores_the_current_day():
    df = add_features(MATCHES, ["E0"])
    assert pd.isna(df.loc[0, "league_btts_trend"])
    assert df.loc[1, "league_btts_trend"] == pytest.approx(1.0)  # only the 2-1 so far
    assert df.loc[3, "league_btts_trend"] == pytest.approx(2 / 3)


def test_league_and_day_columns():
    df = add_features(MATCHES, ["E0", "SP1"])
    assert df.loc[0, "league_E0"] == 1 and df.loc[0, "league_SP1"] == 0
    assert df.loc[0, "day_Saturday"] == 1