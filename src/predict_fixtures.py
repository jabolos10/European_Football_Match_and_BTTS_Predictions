"""
Predict all upcoming matches in our leagues.

Steps:
  1) download the list of upcoming matches from football-data.co.uk
     (fixtures.csv for the big leagues, new_league_fixtures.csv for Ekstraklasa),
  2) compute Elo and all features from the finished matches,
  3) train logistic regression (result and BTTS) and the goals model (expected goals of
     both teams) on all finished matches,
  4) predict every upcoming match and save the table to results/predictions_upcoming.csv.

Run python -m src.load_domestic_leagues first, so the latest results are included.
Run from the project root:  python -m src.predict_fixtures
"""
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest import FIRST_TRAIN_SEASON, LEAGUES, MATCHES_FILE
from src.elo import EloParams, compute_elo
from src.evaluate_models import logistic
from src.features import add_features, feature_groups
from src.goals_model import goals_regressor, markets
from src.load_domestic_leagues import download, read_csv
from src.metrics import market_probabilities, outcome_index

PARAMS_FILE = Path("results/elo_params.json")
OUT_FILE = Path("results/predictions_upcoming.csv")
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
EXTRA_FIXTURES_URL = "https://www.football-data.co.uk/new_league_fixtures.csv"
EXTRA_COUNTRIES = {"Poland": "POL"}  # country name in new_league_fixtures.csv -> our league code
ODDS_COLUMNS = [("AvgH", "AvgD", "AvgA"), ("PSH", "PSD", "PSA"), ("B365H", "B365D", "B365A")]
LEAGUE_NAMES = {"E0": "Premier League", "SP1": "LaLiga", "I1": "Serie A",
                "D1": "Bundesliga", "F1": "Ligue 1", "POL": "Ekstraklasa"}


def season_of(dates):
    """Season = the year it started: matches from July on belong to this year's season."""
    return np.where(dates.dt.month >= 7, dates.dt.year, dates.dt.year - 1)


def pick_odds(df):
    """Average pre-match odds if the file has them, otherwise Pinnacle, otherwise Bet365."""
    df["odds_home"] = df["odds_draw"] = df["odds_away"] = np.nan
    for home, draw, away in ODDS_COLUMNS:
        if {home, draw, away}.issubset(df.columns):
            empty = df["odds_home"].isna()
            df.loc[empty, "odds_home"] = pd.to_numeric(df.loc[empty, home], errors="coerce")
            df.loc[empty, "odds_draw"] = pd.to_numeric(df.loc[empty, draw], errors="coerce")
            df.loc[empty, "odds_away"] = pd.to_numeric(df.loc[empty, away], errors="coerce")
    return df


def load_fixtures():
    """Upcoming matches of our leagues, from today on."""
    main = read_csv(download(FIXTURES_URL, "fixtures.csv", refresh=True))
    main = main[main["Div"].isin(LEAGUES)].rename(
        columns={"Div": "league", "HomeTeam": "home_team", "AwayTeam": "away_team"})

    extra = read_csv(download(EXTRA_FIXTURES_URL, "new_league_fixtures.csv", refresh=True))
    extra = extra[extra["Country"].isin(EXTRA_COUNTRIES)].copy()
    extra["league"] = extra["Country"].map(EXTRA_COUNTRIES)
    extra = extra.rename(columns={"Home": "home_team", "Away": "away_team"})

    fixtures = pick_odds(pd.concat([main, extra], ignore_index=True))
    fixtures["date"] = pd.to_datetime(fixtures["Date"], dayfirst=True, format="mixed")
    fixtures = fixtures[fixtures["date"].dt.date >= date.today()]
    fixtures["season"] = season_of(fixtures["date"])
    fixtures["kick_off"] = fixtures["Time"]
    columns = ["date", "kick_off", "season", "league", "home_team", "away_team",
               "odds_home", "odds_draw", "odds_away"]
    return fixtures[columns].sort_values(["date", "kick_off", "league"]).reset_index(drop=True)


def main():
    params = EloParams(**json.loads(PARAMS_FILE.read_text())) if PARAMS_FILE.exists() else EloParams()
    history, final_ratings = compute_elo(pd.read_csv(MATCHES_FILE), params)
    ratings = dict(zip(final_ratings["team"], final_ratings["elo"]))

    fixtures = load_fixtures()
    unknown = sorted((set(fixtures["home_team"]) | set(fixtures["away_team"])) - set(ratings))
    if unknown:
        print(f"Skipped - teams not in our results data: {unknown}")
        fixtures = fixtures[fixtures["home_team"].isin(ratings) & fixtures["away_team"].isin(ratings)]
    if fixtures.empty:
        raise SystemExit("No upcoming matches found.")
    fixtures["elo_home"] = fixtures["home_team"].map(ratings)
    fixtures["elo_away"] = fixtures["away_team"].map(ratings)
    fixtures["upcoming"] = True

    # Features for finished and upcoming matches together, so upcoming ones see all past results
    history["upcoming"] = False
    combined = pd.concat([history, fixtures], ignore_index=True)
    combined = add_features(combined, LEAGUES)
    columns = [c for group in feature_groups(LEAGUES).values() for c in group]

    train = combined[~combined["upcoming"] & combined["league"].isin(LEAGUES)
                     & (combined["season"] >= FIRST_TRAIN_SEASON)].copy()
    train["outcome"] = outcome_index(train["home_goals"], train["away_goals"])
    train["btts"] = ((train["home_goals"] > 0) & (train["away_goals"] > 0)).astype(int)
    usable = [c for c in columns if train[c].notna().any()]
    result_model = logistic().fit(train[usable], train["outcome"])
    btts_model = logistic().fit(train[usable], train["btts"])

    upcoming = combined[combined["upcoming"]].copy()
    result_probs = np.zeros((len(upcoming), 3))
    result_probs[:, result_model.classes_] = result_model.predict_proba(upcoming[usable])
    upcoming["home_win"], upcoming["draw"], upcoming["away_win"] = result_probs.T
    upcoming["btts_yes"] = btts_model.predict_proba(upcoming[usable])[:, list(btts_model.classes_).index(1)]

    # Goals model: expected goals of both teams -> BTTS and "win + BTTS" from the score table
    home_goals_model = goals_regressor().fit(train[usable], train["home_goals"])
    away_goals_model = goals_regressor().fit(train[usable], train["away_goals"])
    upcoming["xg_home"] = home_goals_model.predict(upcoming[usable])
    upcoming["xg_away"] = away_goals_model.predict(upcoming[usable])
    goals = markets(upcoming["xg_home"], upcoming["xg_away"])
    upcoming["btts_goals_model"] = goals["btts_yes"]
    upcoming["home_win_and_btts"] = goals["home_win_and_btts"]
    upcoming["away_win_and_btts"] = goals["away_win_and_btts"]

    out = upcoming[["date", "kick_off", "league", "home_team", "away_team", "home_win", "draw", "away_win",
                    "btts_yes", "btts_goals_model", "xg_home", "xg_away",
                    "home_win_and_btts", "away_win_and_btts"]].copy()
    out["date"] = out["date"].dt.date
    OUT_FILE.parent.mkdir(exist_ok=True)
    out.round(3).to_csv(OUT_FILE, index=False)

    # Print per league, with the bookmakers' % next to ours (only on screen, not saved)
    market = market_probabilities(upcoming["odds_home"], upcoming["odds_draw"], upcoming["odds_away"])
    out["book_home"], out["book_draw"], out["book_away"] = market.T
    for code in LEAGUES:
        games = out[out["league"] == code]
        if games.empty:
            continue
        print(f"\n{LEAGUE_NAMES[code]}")
        print(f"  {'date':10} {'time':5}  {'match':38} {'home':>5} {'draw':>5} {'away':>5}"
              f"  {'BTTS':>5} {'BTTS*':>5} {'xG':>9}  {'1&BTTS':>6} {'2&BTTS':>6}"
              f"  | bookmakers {'home':>5} {'draw':>5} {'away':>5}")
        for g in games.itertuples():
            match = f"{g.home_team} - {g.away_team}"
            xg = f"{g.xg_home:.1f}-{g.xg_away:.1f}"
            print(f"  {g.date!s:10} {g.kick_off!s:5}  {match:38} {g.home_win:5.0%} {g.draw:5.0%} {g.away_win:5.0%}"
                  f"  {g.btts_yes:5.0%} {g.btts_goals_model:5.0%} {xg:>9}"
                  f"  {g.home_win_and_btts:6.0%} {g.away_win_and_btts:6.0%}"
                  f"  |            {g.book_home:5.0%} {g.book_draw:5.0%} {g.book_away:5.0%}")
    print("\nBTTS = logistic regression, BTTS* = goals model, xG = expected goals home-away,"
          "\n1&BTTS / 2&BTTS = home / away win and both teams score")
    print(f"\nSaved {len(out)} matches to {OUT_FILE}")


if __name__ == "__main__":
    main()
