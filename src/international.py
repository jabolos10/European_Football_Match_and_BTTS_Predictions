"""
National teams: Elo ratings + a goals model for international matches (Nations League, qualifiers, friendlies).

Data: every international match since 1872 from github.com/martj42/international_results
(updated a few days after each international break). It has no fixture list, so the matches
we want to predict are typed into data/international_fixtures.csv by hand before each break.

Model:
  1) Elo for national teams, in the style of the World Football Elo Ratings (eloratings.net):
     bigger tournaments move the ratings more (World Cup K=60 ... friendly K=20), bigger wins
     move them more, the home team gets +100 unless the match is on neutral ground.
  2) Poisson goals model: expected goals of both teams from the Elo difference, home/neutral
     and whether the match is competitive. The score table (with the Dixon-Coles correction)
     gives home/draw/away and both teams to score.

Club data (shots, form, ...) is not used: national teams play too rarely and change squads.

Run from the project root:
  python -m src.international            walk-forward test of the model
  (predictions are made by src/predict_fixtures.py, which calls predict_upcoming())
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor

from src.load_domestic_leagues import download
from src.metrics import accuracy, log_loss, outcome_index, rps
from src.ratings_model import btts_dixon_coles, result_dixon_coles

RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
FIXTURES_FILE = Path("data/international_fixtures.csv")
LEAGUE_CODE = "INT"
FIRST_TRAIN_YEAR = 2010  # the goals model learns from matches since 2010 (Elo itself uses all history)
MIN_TRAIN_ELO = 1600     # ...between teams of at least this level: 8-0 wins over tiny nations would teach the
                         # model that big gaps mean huge wins, and it became overconfident in top-team matches
HOME_ADVANTAGE = 100
START_RATING = 1500
CONTINENTAL_FINALS = ("UEFA Euro", "Copa América", "African Cup of Nations", "AFC Asian Cup", "Gold Cup",
                      "Confederations Cup", "Oceania Nations Cup")


def k_factor(tournament):
    """How much one match can move the ratings: more for bigger tournaments."""
    if tournament == "FIFA World Cup":
        return 60
    if tournament in CONTINENTAL_FINALS:
        return 50
    if "qualification" in tournament or "Nations League" in tournament:
        return 40
    if tournament == "Friendly":
        return 20
    return 30


def goal_factor(goal_difference):
    """A 3-0 win moves the ratings more than a 1-0 win."""
    gd = abs(goal_difference)
    return 1.0 if gd <= 1 else 1.5 if gd == 2 else (11 + gd) / 8


def load_results(refresh=True):
    df = pd.read_csv(download(RESULTS_URL, "international_results.csv", refresh=refresh))
    df = df.rename(columns={"home_score": "home_goals", "away_score": "away_goals"})
    df["neutral"] = df["neutral"].astype(str).str.upper() == "TRUE"
    df["date"] = pd.to_datetime(df["date"])
    return df.dropna(subset=["home_goals", "away_goals"]).sort_values("date", kind="stable").reset_index(drop=True)


def compute_elo(matches):
    """Ratings of both teams BEFORE every match, plus the latest rating of every team."""
    ratings = {}
    before = np.zeros((len(matches), 2))
    for i, m in enumerate(matches[["home_team", "away_team", "home_goals", "away_goals", "tournament", "neutral"]]
                          .itertuples(index=False)):
        home, away = ratings.get(m.home_team, START_RATING), ratings.get(m.away_team, START_RATING)
        before[i] = home, away
        if pd.isna(m.home_goals):
            continue
        gap = home - away + (0 if m.neutral else HOME_ADVANTAGE)
        expected = 1 / (10 ** (-gap / 400) + 1)
        actual = 1.0 if m.home_goals > m.away_goals else 0.5 if m.home_goals == m.away_goals else 0.0
        change = k_factor(m.tournament) * goal_factor(m.home_goals - m.away_goals) * (actual - expected)
        ratings[m.home_team], ratings[m.away_team] = home + change, away - change
    out = matches.copy()
    out["elo_home"], out["elo_away"] = before.T
    return out, ratings


def features(df):
    return pd.DataFrame({
        "elo_diff": (df["elo_home"] - df["elo_away"]) / 100,
        "home": (~df["neutral"]).astype(int),                     # 1 = real home game
        "competitive": (df["tournament"] != "Friendly").astype(int),
    })


def fit(history):
    train = history[(history["elo_home"] >= MIN_TRAIN_ELO) & (history["elo_away"] >= MIN_TRAIN_ELO)]
    x = features(train)
    home_model = PoissonRegressor(alpha=1e-4, max_iter=1000).fit(x, train["home_goals"])
    away_model = PoissonRegressor(alpha=1e-4, max_iter=1000).fit(x, train["away_goals"])
    return home_model, away_model


def predict(models, df):
    x = features(df)
    home_rate, away_rate = models[0].predict(x), models[1].predict(x)
    return home_rate, away_rate, result_dixon_coles(home_rate, away_rate), btts_dixon_coles(home_rate, away_rate)


def load_fixtures(ratings):
    """Hand-typed fixtures (kick-off in Central European Time), with current Elo of both teams."""
    fx = pd.read_csv(FIXTURES_FILE)
    fx["neutral"] = fx["neutral"].astype(str).str.upper() == "TRUE"
    unknown = sorted((set(fx["home_team"]) | set(fx["away_team"])) - set(ratings))
    if unknown:
        print(f"International: unknown team names (check spelling in {FIXTURES_FILE}): {unknown}")
        fx = fx[fx["home_team"].isin(ratings) & fx["away_team"].isin(ratings)]
    fx["elo_home"] = fx["home_team"].map(ratings)
    fx["elo_away"] = fx["away_team"].map(ratings)
    return fx


def predict_upcoming(today):
    """Predictions for the hand-typed fixtures from `today` on, in the same columns as the club predictions."""
    if not FIXTURES_FILE.exists():
        return pd.DataFrame()
    history, ratings = compute_elo(load_results())
    fx = load_fixtures(ratings)
    fx = fx[pd.to_datetime(fx["date"]).dt.date >= today]
    if fx.empty:
        return pd.DataFrame()
    models = fit(history[history["date"].dt.year >= FIRST_TRAIN_YEAR])
    home_rate, away_rate, result, btts = predict(models, fx)

    # the site expects UK kick-off times (like football-data.co.uk); our file has Central European Time
    cet = pd.to_datetime(fx["date"] + " " + fx["kick_off_cet"]).dt.tz_localize("Europe/Warsaw")
    uk = cet.dt.tz_convert("Europe/London")
    out = pd.DataFrame({"date": uk.dt.date, "kick_off": uk.dt.strftime("%H:%M"), "league": LEAGUE_CODE,
                        "home_team": fx["home_team"].to_numpy(), "away_team": fx["away_team"].to_numpy()})
    out["home_win"], out["draw"], out["away_win"] = result.T
    out["btts_yes"] = out["btts_goals_model"] = btts
    # one model for national teams: both website variants show the same numbers
    out["home_win_ratings"], out["draw_ratings"], out["away_win_ratings"] = result.T
    out["btts_ratings_model"] = btts
    out["xg_home"], out["xg_away"] = home_rate, away_rate
    out["home_win_and_btts"] = out["away_win_and_btts"] = np.nan
    return out


def results_for_settling():
    """Finished international matches in the same columns as the club results (date, teams, goals)."""
    df = load_results(refresh=False)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df[["date", "home_team", "away_team", "home_goals", "away_goals"]]


def backtest():
    """Walk-forward: predict each year from 2022 using only matches before that year."""
    history, _ = compute_elo(load_results())
    history["year"] = history["date"].dt.year
    rows = []
    for year in range(2022, history["year"].max() + 1):
        train = history[(history["year"] >= FIRST_TRAIN_YEAR) & (history["year"] < year)]
        test = history[history["year"] == year].copy()
        _, _, result, btts = predict(fit(train), test)
        test[["p_home", "p_draw", "p_away"]] = result
        test["p_btts"] = btts
        test["base_btts"] = ((train["home_goals"] > 0) & (train["away_goals"] > 0)).mean()
        rows.append(test)
    test = pd.concat(rows)
    europe = test["tournament"].str.contains("UEFA") | (test["tournament"] == "FIFA World Cup")
    for name, part in [("all matches", test), ("UEFA competitions + World Cup", test[europe])]:
        outcome = outcome_index(part["home_goals"], part["away_goals"])
        probs = part[["p_home", "p_draw", "p_away"]].to_numpy()
        happened = ((part["home_goals"] > 0) & (part["away_goals"] > 0)).astype(int).to_numpy()
        btts = part["p_btts"].to_numpy()
        base = part["base_btts"].to_numpy()
        print(f"\n{name}: {len(part)} matches, 2022-{test['year'].max()}")
        print(f"  1X2   RPS {rps(probs, outcome):.4f}   favourite right {accuracy(probs, outcome):.1%}")
        print(f"  BTTS  log loss {log_loss(np.column_stack([1 - btts, btts]), happened):.4f}"
              f"   (always the average rate: {log_loss(np.column_stack([1 - base, base]), happened):.4f})"
              f"   real BTTS rate {happened.mean():.1%}")


if __name__ == "__main__":
    backtest()
