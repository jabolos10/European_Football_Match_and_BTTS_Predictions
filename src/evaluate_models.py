"""
Do more features and better models beat plain Elo?

Two questions, both tested the same honest way as src/backtest.py
(for every test season: train on all earlier seasons, predict that season):
  1) match result (home win / draw / away win), scored with RPS - bookmakers are the benchmark,
  2) both teams to score (BTTS yes / no), scored with the Brier score.

Prints:
  - which feature groups help (added one by one, logistic regression),
  - logistic regression vs gradient boosting vs Elo only vs bookmakers,
  - results per league,
  - "confident picks": how often the model is right when it is sure.

Run from the project root:  python -m src.evaluate_models
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.backtest import FIRST_TRAIN_SEASON, LEAGUES, MATCHES_FILE, TEST_SEASONS
from src.elo import EloParams, compute_elo
from src.features import add_features, feature_groups
from src.metrics import accuracy, brier, log_loss, market_probabilities, outcome_index, rps

PARAMS_FILE = Path("results/elo_params.json")
RESULTS_DIR = Path("results")


# ------- models ---------

def base_rates():
    """Always the same %: how often each outcome happened in the training seasons."""
    return DummyClassifier(strategy="prior")


def logistic():
    """Fill missing values with the average, put all features on the same scale, logistic regression."""
    return make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(max_iter=2000))


def boosting():
    """Many small decision trees, each one fixing the mistakes of the previous ones."""
    return HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=300, max_leaf_nodes=15, min_samples_leaf=50,
        l2_regularization=1.0, early_stopping=True, random_state=0)


# ------testing -------

def walk_forward(df, columns, make_model, target, n_classes):
    """For every test season: train on all earlier seasons, predict that season.

    Returns probabilities for every row of df (rows outside the test seasons stay NaN).
    """
    probs = np.full((len(df), n_classes), np.nan)
    for season in TEST_SEASONS:
        train = ((df["season"] >= FIRST_TRAIN_SEASON) & (df["season"] < season)).to_numpy()
        test = (df["season"] == season).to_numpy()
        usable = [c for c in columns if df.loc[train, c].notna().any()]  # skip columns with no data at all
        model = make_model().fit(df.loc[train, usable], df.loc[train, target])
        season_probs = np.zeros((test.sum(), n_classes))
        season_probs[:, model.classes_] = model.predict_proba(df.loc[test, usable])
        probs[test] = season_probs
    return probs


def result_scores(probs, outcomes):
    return {"rps": rps(probs, outcomes), "log_loss": log_loss(probs, outcomes),
            "accuracy": accuracy(probs, outcomes)}


def btts_scores(probs, happened):
    return {"brier": brier(probs[:, 1], happened), "log_loss": log_loss(probs, happened),
            "accuracy": accuracy(probs, happened)}


def print_table(title, rows):
    print(f"\n{title}")
    print(pd.DataFrame(rows).set_index("model").round(4).to_string())


# --------main -------

def load_data():
    params = EloParams(**json.loads(PARAMS_FILE.read_text())) if PARAMS_FILE.exists() else EloParams()
    with_elo, _ = compute_elo(pd.read_csv(MATCHES_FILE), params)
    df = add_features(with_elo, LEAGUES)  # features use matches from ALL leagues (form of promoted teams etc.)
    df = df[df["league"].isin(LEAGUES) & (df["season"] >= FIRST_TRAIN_SEASON)]
    df = df.dropna(subset=["odds_home", "odds_draw", "odds_away"]).reset_index(drop=True)
    df["outcome"] = outcome_index(df["home_goals"], df["away_goals"])
    df["btts"] = ((df["home_goals"] > 0) & (df["away_goals"] > 0)).astype(int)
    return df


def main():
    df = load_data()
    test = df["season"].isin(TEST_SEASONS).to_numpy()
    outcomes, btts = df.loc[test, "outcome"].to_numpy(), df.loc[test, "btts"].to_numpy()
    groups = feature_groups(LEAGUES)
    all_columns = [c for columns in groups.values() for c in columns]
    print(f"{len(df)} matches, {test.sum()} in the test seasons {TEST_SEASONS[0]}-{TEST_SEASONS[-1] + 1}")

    # 1) Which feature groups help? Add them one by one.
    rows, columns = [], []
    for name, group in groups.items():
        columns = columns + group
        result = walk_forward(df, columns, logistic, "outcome", 3)[test]
        both = walk_forward(df, columns, logistic, "btts", 2)[test]
        rows.append({"model": f"+ {name}", "result_rps": rps(result, outcomes),
                     "btts_brier": brier(both[:, 1], btts)})
        print(f"  added {name}")
    print_table("Feature groups added one by one (logistic regression, lower is better):", rows)

    # 2) Match result: models vs bookmakers
    forecasts = {
        "base rates": walk_forward(df, ["elo_diff"], base_rates, "outcome", 3),
        "Elo only": walk_forward(df, ["elo_diff"], logistic, "outcome", 3),
        "logistic, all features": walk_forward(df, all_columns, logistic, "outcome", 3),
        "boosting, all features": walk_forward(df, all_columns, boosting, "outcome", 3),
    }
    forecasts = {name: probs[test] for name, probs in forecasts.items()}
    tested = df[test]
    forecasts["bookmakers"] = market_probabilities(tested["odds_home"], tested["odds_draw"], tested["odds_away"])
    print_table("Match result (home / draw / away):",
                [{"model": name, **result_scores(p, outcomes)} for name, p in forecasts.items()])

    # 3) BTTS: models (no bookmaker BTTS odds in our data yet)
    btts_forecasts = {
        "base rates": walk_forward(df, ["elo_diff"], base_rates, "btts", 2)[test],
        "logistic, all features": walk_forward(df, all_columns, logistic, "btts", 2)[test],
        "boosting, all features": walk_forward(df, all_columns, boosting, "btts", 2)[test],
    }
    print_table("Both teams to score (yes / no):",
                [{"model": name, **btts_scores(p, btts)} for name, p in btts_forecasts.items()])

    # 4) Per league
    leagues = tested["league"].to_numpy()
    per_league = []
    for code in LEAGUES:
        mask = leagues == code
        row = {"model": code}
        for name in ["Elo only", "logistic, all features", "boosting, all features", "bookmakers"]:
            row[f"rps: {name}"] = rps(forecasts[name][mask], outcomes[mask])
        row["btts brier: logistic"] = brier(btts_forecasts["logistic, all features"][mask, 1], btts[mask])
        row["btts rate"] = btts[mask].mean()
        per_league.append(row)
    print_table("Per league:", per_league)

        # 5) Start of the season (after summer transfers) vs the rest
    early = tested["home_season_games"].to_numpy() < 5
    rows = []
    for name in ["Elo only", "logistic, all features", "bookmakers"]:
        rows.append({"model": name,
                     "rps: first 5 rounds": rps(forecasts[name][early], outcomes[early]),
                     "rps: rest of season": rps(forecasts[name][~early], outcomes[~early])})
    print_table(f"Start of the season ({early.sum()} matches) vs the rest ({(~early).sum()} matches):", rows)

    # 6) Confident picks: when the forecast is sure, how often is it right?
    rows = []
    for threshold in [0.5, 0.6, 0.7]:
        for name in ["logistic, all features", "bookmakers"]:
            sure = forecasts[name].max(axis=1) >= threshold
            rows.append({"model": f"{name}, favourite >= {threshold:.0%}", "matches": int(sure.sum()),
                         "accuracy": accuracy(forecasts[name][sure], outcomes[sure])})
        yes = btts_forecasts["logistic, all features"][:, 1] >= threshold
        rows.append({"model": f"BTTS yes >= {threshold:.0%} (logistic)", "matches": int(yes.sum()),
                     "accuracy": btts[yes].mean() if yes.any() else np.nan})
    print_table("Confident picks:", rows)

    RESULTS_DIR.mkdir(exist_ok=True)
    pd.DataFrame([{"model": n, **result_scores(p, outcomes)} for n, p in forecasts.items()]).to_csv(
        RESULTS_DIR / "model_comparison.csv", index=False)
    print(f"\nSaved {RESULTS_DIR / 'model_comparison.csv'}")


if __name__ == "__main__":
    main()