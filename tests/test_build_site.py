"""
Tests for src/build_site.py - mainly: a prediction can never change after its match day has started.

Run from the project root:  python -m pytest -q
"""
from datetime import datetime, timezone

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
