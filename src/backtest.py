"""
Backtest: how good are our Elo-based predictions compared with the bookmakers?

Steps:
1) choose the Elo settings (K, home_advantage, start of promoted teams) on two validation seasons, 2019-20 and 2020-21,
2) test on 2021-2022 ... 2025-26: for every test season we train on all earlier seasons and predict that season (the model never sees the season it predicts),
3) compare three forecasters on the same matches:
    - base rates: always the same %, how often home win / draw / away win happened before,
    - Elo model: logistic regression on the Elo difference of the two teams,
    - bookmakers: closing odds turned into probabilities.

Saves the chosen Elo settings and the test results to results/.
Run from the project root: python -m src.backtest
"""
import itertools
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.elo import EloParams, compute_elo
from src.metrics import accuracy, log_loss, market_probabilities, outcome_index, rps

MATCHES_FILE = "data/processed/domestic_matches.csv"
RESULT_DIR = Path("results")
LEAGUES = ["E0", "SP1", "I1", "D1", "F1", "POL"]  # top 5 leagues + Ekstraklasa
FIRST_TRAIN_SEASON = 2017   #2016-17 is only a warm-up season for the ratings
VALIDATION_SEASONS = [2019, 2020]
TEST_SEASONS = [2021, 2022, 2023, 2024, 2025]
GRID = {
    "k": [8, 10, 12, 15, 20],
    "home_advantage": [40, 60, 80],
    "new_team_gap": [50, 100, 150],
}

def prepare(matches, params):
    """Elo for all leagues, then keep only our leagues and add the model input and the result."""
    with_elo, _ = compute_elo(matches, params)
    df = with_elo[with_elo["league"].isin(LEAGUES)].copy()
    df["elo_diff"] = (df["elo_home"] - df["elo_away"]) / 100    #scaled: 1.0 = 100 Elo points
    df["outcome"] = outcome_index(df["home_goals"], df["away_goals"])
    return df

def fit_model(train):
    """Logistic regression: Elo difference -> probabilities of home win / draw / away win."""
    model = LogisticRegression()
    model.fit(train[["elo_diff"]], train["outcome"])
    return model

def predict_probs(model, df):
    """Probabilities in the order home win / draw / away win."""
    probs = model.predict_proba(df[["elo_diff"]])
    ordered = np.zeros((len(df), 3))
    ordered[:, model.classes_] = probs
    return ordered

def evaluate_season(df, season):
    """Train on all seasons before `season`, test on `season`.

    One model is trained on all our leagues together; scores are reported per league.
    Returns one row of scores per league and forecaster.
    """
    train = df[(df["season"] >= FIRST_TRAIN_SEASON) & (df["season"] < season)]
    test_all = df[df["season"] == season].dropna(subset=["odds_home", "odds_draw", "odds_away"])
    model = fit_model(train)
    base_rates = np.bincount(train["outcome"], minlength=3) / len(train)

    rows = []
    for league, test in test_all.groupby("league"):
        outcomes = test["outcome"].to_numpy()
        forecasts = {
            "base rates": np.tile(base_rates, (len(test), 1)),
            "Elo model": predict_probs(model, test),
            "bookmakers": market_probabilities(test["odds_home"], test["odds_draw"], test["odds_away"]),
        }
        rows += [
            {"season": season, "league": league, "forecaster": name, "matches": len(test),
             "rps": rps(probs, outcomes), "log_loss": log_loss(probs, outcomes),
             "accuracy": accuracy(probs, outcomes)}
            for name, probs in forecasts.items()
        ]
    return rows

def choose_elo_params(matches):
    """Try every combination in GRID, keep the one with the lowest RPS,on the validation seasons."""
    results = []
    combos = list(itertools.product(*GRID.values()))
    for i, values in enumerate(combos, start=1):
        params = EloParams(**dict(zip(GRID.keys(), values)))
        df = prepare(matches, params)
        rows = [r for s in VALIDATION_SEASONS for r in evaluate_season(df, s) if r["forecaster"] == "Elo model"]
        score = np.average([r["rps"] for r in rows], weights=[r["matches"] for r in rows])
        results.append({**dict(zip(GRID.keys(), values)), "validation_rps": score})
        print(f"  [{i}/{len(combos)}] {dict(zip(GRID.keys(), values))} -> RPS {score:.4f}")
    table = pd.DataFrame(results).sort_values("validation_rps", ignore_index=True)
    best = table.iloc[0]
    return EloParams(**{key: float(best[key]) for key in GRID}), table


def main():
    matches = pd.read_csv(MATCHES_FILE)

    print("Choosing Elo settings on the validation seasons...")
    params, tuning = choose_elo_params(matches)
    print("\nBest 5 settings:")
    print(tuning.head(5).round(4).to_string(index=False))

    print(f"\nTesting on seasons {TEST_SEASONS[0]}-{TEST_SEASONS[-1] + 1} with {asdict(params)}")
    df = prepare(matches, params)
    rows = [r for season in TEST_SEASONS for r in evaluate_season(df, season)]
    results = pd.DataFrame(rows)

    def weighted(group):
        """Average scores over seasons/leagues, weighted by the number of matches."""
        return pd.Series({
            "matches": group["matches"].sum(),
            "rps": np.average(group["rps"], weights=group["matches"]),
            "log_loss": np.average(group["log_loss"], weights=group["matches"]),
            "accuracy": np.average(group["accuracy"], weights=group["matches"]),
        })

    per_league = results.groupby(["league", "forecaster"]).apply(weighted, include_groups=False)
    overall = results.groupby("forecaster").apply(weighted, include_groups=False).sort_values("rps")

    print("\nRPS per league, all test seasons (lower is better):")
    print(per_league["rps"].unstack().round(4).to_string())
    print("\nAll leagues and test seasons together:")
    print(overall.round(4).to_string())

    RESULT_DIR.mkdir(exist_ok=True)
    (RESULT_DIR / "elo_params.json").write_text(json.dumps(asdict(params), indent=2))
    results.to_csv(RESULT_DIR / "backtest_results.csv", index=False)
    print(f"\nSaved {RESULT_DIR / 'elo_params.json'} and {RESULT_DIR / 'backtest_results.csv'}")


if __name__ == "__main__":
    main()