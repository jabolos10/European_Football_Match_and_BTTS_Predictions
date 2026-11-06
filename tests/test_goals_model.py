"""
Tests for src/goals_model.py.

Run from the project root:  python -m pytest -q
"""
import numpy as np
import pytest

from src.goals_model import markets, score_table


def test_score_table_adds_up_to_one():
    table = score_table([1.5, 0.8], [1.1, 2.0])
    assert table.sum(axis=(1, 2)) == pytest.approx([1.0, 1.0])


def test_btts_matches_the_poisson_formula():
    # P(both score) = P(home scores at least 1) * P(away scores at least 1)
    result = markets([1.5], [1.1])
    expected = (1 - np.exp(-1.5)) * (1 - np.exp(-1.1))
    assert result["btts_yes"][0] == pytest.approx(expected, abs=1e-6)


def test_result_probabilities_add_up_and_favour_the_stronger_team():
    result = markets([2.0], [0.7])
    total = result["home_win"] + result["draw"] + result["away_win"]
    assert total[0] == pytest.approx(1.0)
    assert result["home_win"][0] > result["away_win"][0]


def test_combination_is_smaller_than_each_part():
    result = markets([1.6], [1.2])
    assert result["home_win_and_btts"][0] < min(result["home_win"][0], result["btts_yes"][0])
