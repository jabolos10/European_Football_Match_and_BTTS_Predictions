"""
Tests for src/metrics.py.
 
Run from the project root:  python -m pytest -q
"""
import numpy as np
import pytest
 
from src.metrics import accuracy, log_loss, market_probabilities, outcome_index, rps
 
 
def test_outcome_index():
    assert outcome_index([2, 1, 0], [1, 1, 3]).tolist() == [0, 1, 2]
 
 
def test_perfect_forecast_has_zero_rps():
    assert rps([[1, 0, 0], [0, 0, 1]], [0, 2]) == pytest.approx(0.0)
 
 
def test_rps_known_value():
    # forecast 50/30/20, home team won: ((0.5 - 1)^2 + (0.8 - 1)^2) / 2 = 0.145
    assert rps([[0.5, 0.3, 0.2]], [0]) == pytest.approx(0.145)
 
 
def test_rps_punishes_the_far_outcome_more():
    """When the home team wins, 40% on a draw is less wrong than 40% on an away win."""
    near = rps([[0.6, 0.4, 0.0]], [0])
    far = rps([[0.6, 0.0, 0.4]], [0])
    assert near < far
 
 
def test_log_loss_and_accuracy():
    probs = [[0.5, 0.3, 0.2], [0.2, 0.3, 0.5]]
    assert log_loss(probs, [0, 2]) == pytest.approx(-np.log(0.5))
    assert accuracy(probs, [0, 1]) == pytest.approx(0.5)
 
 
def test_market_probabilities_remove_the_margin():
    probs = market_probabilities([2.0], [3.4], [3.8])
    assert probs.sum() == pytest.approx(1.0)
    assert probs[0, 0] > probs[0, 1]
 