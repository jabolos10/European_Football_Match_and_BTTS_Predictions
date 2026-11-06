"""
Predict one match with the Elo model.
 
Example (team names as in football-data.co.uk):
    python -m src.predict "Ath Madrid" "Barcelona"
 
Uses the Elo settings chosen by src.backtest (results/elo_params.json),
trains the logistic regression on every finished match from FIRST_TRAIN_SEASON on
and uses both teams' current ratings.
"""
import argparse
import difflib
import json
from pathlib import Path
 
import pandas as pd
 
from src.backtest import FIRST_TRAIN_SEASON, LEAGUES, MATCHES_FILE, fit_model, predict_probs, prepare
from src.elo import EloParams, compute_elo
 
PARAMS_FILE = Path("results/elo_params.json")
 
 
def load_params():
    if PARAMS_FILE.exists():
        return EloParams(**json.loads(PARAMS_FILE.read_text()))
    print(f"{PARAMS_FILE} not found, using default Elo settings (run python -m src.backtest first)")
    return EloParams()
 
 
def check_team(team, matches):
    """Stop with a list of valid names if the team is not in the data."""
    current = matches[(matches["season"] == matches["season"].max()) & matches["league"].isin(LEAGUES)]
    valid = sorted(set(current["home_team"]) | set(current["away_team"]))
    if team not in set(matches["home_team"]) | set(matches["away_team"]):
        close = difflib.get_close_matches(team, valid, n=3, cutoff=0.6)
        hint = f"Did you mean: {close}?" if close else "Names in this season's Premier League and LaLiga:"
        raise SystemExit(f"Unknown team '{team}'. {hint}\n" + ", ".join(valid))
 
 
def main():
    parser = argparse.ArgumentParser(description="Predict one match.")
    parser.add_argument("home_team")
    parser.add_argument("away_team")
    args = parser.parse_args()
 
    matches = pd.read_csv(MATCHES_FILE)
    params = load_params()
 
    _, final_ratings = compute_elo(matches, params)
    ratings = dict(zip(final_ratings["team"], final_ratings["elo"]))
    for team in (args.home_team, args.away_team):
        check_team(team, matches)
 
    df = prepare(matches, params)
    model = fit_model(df[df["season"] >= FIRST_TRAIN_SEASON])
 
    match = pd.DataFrame({"elo_diff": [(ratings[args.home_team] - ratings[args.away_team]) / 100]})
    home_win, draw, away_win = predict_probs(model, match)[0]
 
    print(f"\n{args.home_team} (Elo {ratings[args.home_team]:.0f}) vs "
          f"{args.away_team} (Elo {ratings[args.away_team]:.0f})")
    print(f"  Home win: {home_win:6.1%}")
    print(f"  Draw:     {draw:6.1%}")
    print(f"  Away win: {away_win:6.1%}")
 
 
if __name__ == "__main__":
    main()
 