"""Quick check that our Elo ratings look like real football.

Prints:
  1) the current top 10 of the Premier League and LaLiga by our Elo,
  2) how often the Elo favourite won vs how often the bookmakers' favourite won
     (matches that did not end in a draw, from CHECK_FROM_SEASON onwards).

Run from the project root:  python -m src.show_elo
"""
import pandas as pd

from src.elo import EloParams, compute_elo

MATCHES_FILE = "data/processed/domestic_matches.csv"
LEAGUES = {"E0": "Premier League", "SP1": "LaLiga", "I1": "Serie A", "D1": "Bundesliga", "F1": "Ligue 1", "POL": "Ekstraklasa"}
CHECK_FROM_SEASON = 2021


def season_label(start_year):
    """2026 -> '2026/27'."""
    return f"{start_year}/{(start_year + 1) % 100:02d}"


def main():
    matches = pd.read_csv(MATCHES_FILE)
    params = EloParams()
    with_elo, final_ratings = compute_elo(matches, params)
    current_season = with_elo["season"].max()

    # 1) Current top 10 in each league
    for code, name in LEAGUES.items():
        this_season = with_elo[(with_elo["league"] == code) & (with_elo["season"] == current_season)]
        teams_now = set(this_season["home_team"]) | set(this_season["away_team"])
        table = final_ratings[final_ratings["team"].isin(teams_now)].head(10).reset_index(drop=True)
        table.index += 1
        print(f"\n{name} - top 10 by our Elo ({season_label(current_season)}):")
        print(table[["team", "elo"]].round(0).to_string())

    # 2) Who picks the winner more often: our Elo or the bookmakers?
    recent = with_elo[with_elo["league"].isin(list(LEAGUES)) & (with_elo["season"] >= CHECK_FROM_SEASON)]
    decisive = recent[recent["home_goals"] != recent["away_goals"]].dropna(subset=["odds_home", "odds_away"])
    home_won = decisive["home_goals"] > decisive["away_goals"]
    elo_picks_home = decisive["elo_home"] + params.home_advantage > decisive["elo_away"]
    market_picks_home = decisive["odds_home"] < decisive["odds_away"]

    print(f"\nPremier League + LaLiga since {season_label(CHECK_FROM_SEASON)}, "
          f"{len(decisive)} matches without a draw:")
    print(f"  Elo favourite won:        {(elo_picks_home == home_won).mean():.1%}")
    print(f"  Bookmaker favourite won:  {(market_picks_home == home_won).mean():.1%}")


if __name__ == "__main__":
    main()
