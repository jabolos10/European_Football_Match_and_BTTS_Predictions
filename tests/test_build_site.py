"""
Tests for src/build_site.py - mainly: a prediction can never change after its match day has started.

Run from the project root:  python -m pytest -q
"""
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import src.build_site as site

COLUMNS = ["date", "kick_off", "league", "home_team", "away_team", "home_win", "draw", "away_win",
           "btts_yes", "btts_goals_model", "xg_home", "xg_away", "home_win_and_btts", "away_win_and_btts"]


def write_predictions(path, rows):
    pd.DataFrame(rows, columns=COLUMNS).to_csv(path, index=False)


def row(date, home_win):
    return [date, "15:00", "E0", "A", "B", home_win, 0.3, 0.7 - home_win, 0.5, 0.5, 1.4, 1.1, 0.2, 0.1]


def test_prediction_is_frozen_once_match_day_starts(tmp_path, monkeypatch):
    monkeypatch.setattr(site, "PREDICTIONS_FILE", tmp_path / "upcoming.csv")
    monkeypatch.setattr(site, "LOG_FILE", tmp_path / "log.csv")

    write_predictions(site.PREDICTIONS_FILE, [row("2026-11-08", 0.40)])
    site.update_log(datetime(2026, 11, 7, 5, tzinfo=timezone.utc))      # day before: logged

    write_predictions(site.PREDICTIONS_FILE, [row("2026-11-08", 0.45)])
    site.update_log(datetime(2026, 11, 8, 5, tzinfo=timezone.utc))      # match day morning: may still change

    write_predictions(site.PREDICTIONS_FILE, [row("2026-11-08", 0.90)])
    log = site.update_log(datetime(2026, 11, 8, 18, tzinfo=timezone.utc))  # evening: too late, ignored

    assert len(log) == 1
    assert log.loc[0, "home_win"] == 0.45


def test_settle_matches_predictions_with_results(tmp_path, monkeypatch):
    monkeypatch.setattr(site, "MATCHES_FILE", tmp_path / "matches.csv")
    pd.DataFrame({"date": ["2026-11-08"], "home_team": ["A"], "away_team": ["B"],
                  "home_goals": [2], "away_goals": [1]}).to_csv(site.MATCHES_FILE, index=False)
    log = pd.DataFrame([row("2026-11-08", 0.5), row("2026-11-15", 0.5)], columns=COLUMNS)
    settled = site.settle(log)
    assert len(settled) == 1              # the second match is not played yet
    assert bool(settled.loc[0, "pick_right"])
    assert settled.loc[0, "btts"] == 1


def test_badge_uses_crest_with_initials_fallback(monkeypatch):
    from src import build_site
    monkeypatch.setattr(build_site, "CRESTS", {"Man City": "https://crests.football-data.org/65.png"})
    with_crest = build_site.badge("Man City")
    assert '<img class="crest-img" src="https://crests.football-data.org/65.png"' in with_crest
    assert '<span class="crest"' in with_crest and " hidden>" in with_crest  # fallback if the image fails
    assert build_site.badge("Legia") == build_site.initials_badge("Legia")  # no crest -> initials only


def test_model_duel_compares_only_matches_both_models_predicted():
    settled = pd.DataFrame({
        "league": ["E0", "E0", "POL", "SP1"],
        "outcome": [0, 1, 2, 0],          # home win, draw, away win, home win
        "btts": [1, 0, 1, 1],
        "home_win": [0.5, 0.5, 0.2, 0.6], "draw": [0.3, 0.3, 0.3, 0.2], "away_win": [0.2, 0.2, 0.5, 0.2],
        "home_win_ratings": [0.6, 0.3, 0.5, np.nan], "draw_ratings": [0.2, 0.4, 0.3, np.nan],
        "away_win_ratings": [0.2, 0.3, 0.2, np.nan],
        "btts_goals_model": [0.7, 0.6, 0.4, 0.65],
        "btts_ratings_model": [0.65, 0.3, 0.55, np.nan],   # last match: logged before the ratings model existed
    })
    duel = site.model_duel(settled)
    assert duel["matches"] == 3
    assert duel["goals"]["favourite"] == 2 / 3 and duel["ratings"]["favourite"] == 2 / 3
    assert duel["goals"]["right"] == 1 / 3      # BTTS: 0.7 yes ✓, 0.6 yes ✗, 0.4 no ✗
    assert duel["ratings"]["right"] == 1        # all three BTTS calls right
    assert duel["ratings"]["strong_n"] == 1 and duel["goals"]["strong_n"] == 2
    assert "Model duel" in site.duel_html(duel)
    assert site.model_duel(settled.iloc[3:]) is None
