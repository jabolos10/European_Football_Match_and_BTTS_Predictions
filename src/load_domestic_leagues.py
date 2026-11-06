"""Download domestic league results from football-data.co.uk and put them in one CSV.

We use these matches for:
  1) our own team ratings (Elo now, pi-ratings / Kalman filter later),
  2) the Premier League and LaLiga models (first stage of the project),
  3) later the Champions League model (its teams come from many leagues).

For every match we keep: date, season, league, teams, goals, shots on target
(big leagues only) and 1X2 betting odds - closing odds when the file has them.
The odds are only a benchmark ("how close are we to the bookmakers?"),
never an input to our models.

Output: data/processed/domestic_matches.csv
Run from the project root:  python -m src.load_domestic_leagues
"""
from pathlib import Path

import pandas as pd
import requests

FIRST_SEASON = 2016  # 2016-17: gives the ratings time to "warm up" before the seasons we test on
LAST_SEASON = 2026   # 2026-27: the current season

# Big leagues: one file per season, e.g. https://www.football-data.co.uk/mmz4281/2526/E0.csv
MAIN_LEAGUES = {
    "ENG": "E0", "ESP": "SP1", "GER": "D1", "ITA": "I1", "FRA": "F1", "NED": "N1",
    "POR": "P1", "BEL": "B1", "SCO": "SC0", "TUR": "T1", "GRE": "G1",
}
MAIN_URL = "https://www.football-data.co.uk/mmz4281/{code}/{league}.csv"

# Smaller leagues: one file with all seasons, e.g. https://www.football-data.co.uk/new/AUT.csv
EXTRA_LEAGUES = {
    "AUT": "AUT", "SUI": "SWZ", "DEN": "DNK", "NOR": "NOR",
    "SWE": "SWE", "RUS": "RUS", "POL": "POL", "ROU": "ROU",
}
EXTRA_URL = "https://www.football-data.co.uk/new/{league}.csv"

RAW_DIR = Path("data/raw/football_data")
OUT_FILE = Path("data/processed/domestic_matches.csv")

# 1X2 odds columns, best first. Closing odds (the last price before kick-off) are the
# strongest benchmark, so we use them whenever a file has them.
ODDS_SOURCES = [
    ("pinnacle_closing", "PSCH", "PSCD", "PSCA"),
    ("average_closing", "AvgCH", "AvgCD", "AvgCA"),
    ("pinnacle", "PSH", "PSD", "PSA"),
    ("average", "AvgH", "AvgD", "AvgA"),
    ("betbrain_average", "BbAvH", "BbAvD", "BbAvA"),
]

OUTPUT_COLUMNS = [
    "date", "season", "country", "league", "home_team", "away_team",
    "home_goals", "away_goals", "home_shots_target", "away_shots_target",
    "home_shots", "away_shots",
    "odds_home", "odds_draw", "odds_away", "odds_source",
]


def season_code(start_year):
    """2025 -> '2526' (the format football-data.co.uk uses in its URLs)."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def download(url, filename, refresh=False):
    """Download a file and keep it in data/raw/football_data/. Returns None if it doesn't exist.

    Old seasons never change, so they are downloaded only once. Files that still change
    (the current season) are downloaded again every time with refresh=True.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / filename
    if refresh or not path.exists():
        response = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        if response.status_code == 404:
            return None
        response.raise_for_status()
        path.write_bytes(response.content)
    return path


def read_csv(path):
    """Read a football-data.co.uk file (most are UTF-8, some older ones are latin-1)."""
    try:
        return pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="warn")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1", on_bad_lines="warn")


def add_odds(df):
    """Fill odds_home / odds_draw / odds_away from the best odds columns available.

    Goes through ODDS_SOURCES from best to worst and only fills rows that are still empty,
    so every match gets the best odds its file has. odds_source says which ones were used.
    """
    df["odds_home"] = float("nan")
    df["odds_draw"] = float("nan")
    df["odds_away"] = float("nan")
    df["odds_source"] = None
    for name, home_col, draw_col, away_col in ODDS_SOURCES:
        if not {home_col, draw_col, away_col}.issubset(df.columns):
            continue
        home = pd.to_numeric(df[home_col], errors="coerce")
        draw = pd.to_numeric(df[draw_col], errors="coerce")
        away = pd.to_numeric(df[away_col], errors="coerce")
        fill = df["odds_home"].isna() & home.notna() & draw.notna() & away.notna()
        df.loc[fill, "odds_home"] = home[fill]
        df.loc[fill, "odds_draw"] = draw[fill]
        df.loc[fill, "odds_away"] = away[fill]
        df.loc[fill, "odds_source"] = name
    return df


def finish(df, country, league):
    """Add odds and the columns every file must have, keep only the columns we use."""
    df = add_odds(df)
    for column in ["home_shots", "away_shots", "home_shots_target", "away_shots_target"]:
        if column not in df.columns:
            df[column] = float("nan")  # smaller leagues have no shot statistics
    df["country"], df["league"] = country, league
    return df[[c for c in OUTPUT_COLUMNS if c != "date"] + ["Date"]]


def load_main_league(country, league):
    frames = []
    for year in range(FIRST_SEASON, LAST_SEASON + 1):
        code = season_code(year)
        path = download(MAIN_URL.format(code=code, league=league), f"{league}_{code}.csv",
                        refresh=(year == LAST_SEASON))
        if path is None:
            print(f"  {league} {code}: not found, skipped")
            continue
        df = read_csv(path).rename(columns={
            "HomeTeam": "home_team", "AwayTeam": "away_team",
            "FTHG": "home_goals", "FTAG": "away_goals",
            "HST": "home_shots_target", "AST": "away_shots_target",
            "HS": "home_shots", "AS": "away_shots",
        })
        df["season"] = year
        frames.append(finish(df, country, league))
    return pd.concat(frames, ignore_index=True)


def load_extra_league(country, league):
    path = download(EXTRA_URL.format(league=league), f"{league}.csv", refresh=True)  # includes the current season
    if path is None:
        print(f"  {league}: not found, skipped")
        return pd.DataFrame()
    df = read_csv(path).rename(columns={
        "Home": "home_team", "Away": "away_team",
        "HG": "home_goals", "AG": "away_goals",
    })
    # "2024/2025" -> 2024, and "2025" (leagues played in one calendar year) -> 2025
    df["season"] = pd.to_numeric(df["Season"].astype(str).str[:4], errors="coerce")
    return finish(df, country, league)


def main():
    frames = []
    for country, league in MAIN_LEAGUES.items():
        print(f"{country} ({league})")
        frames.append(load_main_league(country, league))
    for country, league in EXTRA_LEAGUES.items():
        print(f"{country} ({league})")
        frames.append(load_extra_league(country, league))

    df = pd.concat([f for f in frames if not f.empty], ignore_index=True)

    # Drop empty rows and matches without a score (not played yet)
    df = df.dropna(subset=["home_team", "away_team", "home_goals", "away_goals", "season"])
    df = df[df["season"] >= FIRST_SEASON]

    # Dates are written as day/month/year, sometimes with a 2-digit year
    df["date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed").dt.date
    df["season"] = df["season"].astype(int)
    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)
    df = df[OUTPUT_COLUMNS].sort_values("date", kind="stable").reset_index(drop=True)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_FILE, index=False)

    summary = df.groupby(["country", "league"]).agg(
        matches=("home_team", "size"),
        seasons=("season", "nunique"),
        with_odds=("odds_home", lambda s: f"{s.notna().mean():.0%}"),
    )
    print(f"\nSaved {len(df)} matches to {OUT_FILE}\n")
    print(summary.to_string())


if __name__ == "__main__":
    main()
