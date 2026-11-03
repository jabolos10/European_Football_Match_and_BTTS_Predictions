"""Downloads Champions League results from openfootbal and turns them into one CSV.

Output: data/processed/ucl_matches.csv (one row per match, score after 90 minutes).
Run from the project root: python -m src.load_ucl_history
"""
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
URL = "https://raw.githubusercontent.com/openfootball/champions-league/master/{season}/cl.txt"

RAW_DIR = Path("data/raw/openfootball")
OUT_FILE = Path("data/processed/ucl_matches.csv")

# "Tue Sep 16 2025" or "Wed Sep 17" (the year is sometimes missing so we work it out ourselves)
DATE_RE = re.compile(r"^\s+\w{3} (?P<month>\w{3}) (?P<day>\d{1,2})(?: (?P<year>\d{4}))?\s*$")

# "18:45 Athletic Club (ESP) vs Arsenal FC (ENG) 0-2 (0-0)" (kick-off time is optional)
MATCH_RE = re.compile(
    r"^\s+(?:\d{2}:\d{2}\s+)?"
    r"(?P<home>.+?) \((?P<home_country>[A-Z]{3})\)\s+v\s+"
    r"(?P<away>.+?) \((?P<away_country>[A-Z]{3})\)\s+"
    r"(?P<score>\d+-\d+.*)$"
)


def download_season(season):
    """Downloads one season file (only if we do not have it yet) and returns its text."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{season}.txt"
    if not path.exists():
        response = requests.get(URL.format(season=season), timeout=30)
        response.raise_for_status()
        path.write_text(response.text, encoding="utf-8")
    
    return path.read_text(encoding="utf-8")

def parse_score(score_text):
    """Returns (home_goals, away_goals) after 90 minutes, plus extra time / penalties flags.
    
    Score format: 
    "<final score>  (<half-time score)"  ->  normal match 90'
    "<score after ET>  a.e.t. (<90' score>, <half-time score>)"   -> extra time 
    "<penalties> pen. <score after ET> a.e.t. (<90' score>, <half-time score>)"  -> penalties
    
    Examples:
    "0-2 (0-0)"    -> normal match 90' score is 0-2, hal-time score (0-0)
    "3-2 a.e.t. (3-0, 1-0)"     -> extra time, 90' score is first in brackets: (3-0), half-time score is second (1-0)
    "4-3 pen. 1-1 a.e.t. (1-1, 0-1)"  -> penalties, a.e.t. score is 1-1, 90' score is (1-1), half-time score is (0-1)
    """
    extra_time = "a.e.t." in score_text
    penalties = "pen." in score_text
    if extra_time:
        score_90 = re.search(r"\((\d+)-(\d+)", score_text)
    else:
        score_90 = re.match(r"(\d+)-(\d+)", score_text)

    return int(score_90.group(1)), int(score_90.group(2)), extra_time, penalties

def parse_season(text, season):
    #Go through the file line by line and collect matches
    rows = []
    stage = None
    start_year = int(season[:4])
    current_date = None

    for line in text.splitlines():
        if line.startswith("▪"):
            stage = line.lstrip("▪ ").strip()
            continue

        date_match = DATE_RE.match(line)
        if date_match:
            month = datetime.strptime(date_match["month"], "%b").month
            # A season runs from summer to spring: Jul- Dec = first year, Jan- Jun = second year

            year = start_year if month >= 7 else start_year + 1
            current_date = datetime(year, month, int(date_match["day"])).date()
            continue

        match = MATCH_RE.match(line)
        if match:
            home_goals, away_goals, extra_time, penalties = parse_score(match["score"])
            rows.append({
                "season": season,
                "date": current_date,
                "stage": stage,
                "home_team": match["home"].strip(),
                "home_country": match["home_country"],
                "away_team": match["away"].strip(),
                "away_country": match["away_country"],
                "home_goals": home_goals,
                "away_goals": away_goals,
                "extra_time": extra_time,
                "penalties": penalties,
            })

    return rows



def main():
    all_rows = []
    for season in SEASONS:
        text = download_season(season)
        rows = parse_season(text, season)
        print(f"{season}: {len(rows)} matches")
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)

    #Targets we want to predict (based on the 90-minute score)
    df["result"] = "D"
    df.loc[df["home_goals"] > df["away_goals"], "result"] = "H"
    df.loc[df["home_goals"] < df["away_goals"], "result"] = "A"
    df["btts"] = ((df["home_goals"] > 0) & (df["away_goals"] > 0)).astype(int)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_FILE, index=False)
    print(f"Saved {len(df)} matches to {OUT_FILE}")



if __name__ == "__main__":
    main()
