"""
Download club crest links from football-data.org and match them to our team names.

Only the top 5 leagues - Ekstraklasa is not in the football-data.org free plan, so its
teams keep the coloured initials badge on the site.

Our names come from football-data.co.uk ("Man City", "Ath Madrid"), football-data.org uses
full names ("Manchester City FC", "Club Atlético de Madrid"). We compare every pair of names
in a league, score how similar they are, and pick the best one-to-one pairing
(every team gets exactly one crest, no crest is used twice).

The result is data/crests.csv - check it once, fix a wrong row by hand if needed.
The site only reads that file; it links to the crest images, it does not copy them.

Needs FOOTBALL_DATA_API_KEY in .env (free key from football-data.org).
Run from the project root:  python -m src.fetch_crests
"""
import difflib
import os
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.optimize import linear_sum_assignment

MATCHES_FILE = Path("data/processed/domestic_matches.csv")
OUT_FILE = Path("data/crests.csv")
API_URL = "https://api.football-data.org/v4/competitions/{code}/teams"
COMPETITIONS = {"E0": "PL", "SP1": "PD", "I1": "SA", "D1": "BL1", "F1": "FL1"}  # our code -> football-data.org code
CHECK_BELOW = 0.6  # pairs with a lower similarity are printed for a manual check

# Abbreviations football-data.co.uk uses -> words used in full club names
ABBREVIATIONS = {
    "man": "manchester", "ath": "athletic atletico", "nott'm": "nottingham", "sp": "sporting",
    "st": "saint", "ein": "eintracht", "m'gladbach": "monchengladbach", "espanol": "espanyol",
    "sociedad": "real sociedad", "vallecano": "rayo vallecano", "betis": "real betis", "psg": "paris saint germain",
    "inter": "internazionale", "wolves": "wolverhampton", "spurs": "tottenham", "koln": "koln cologne",
}
NOISE = {"fc", "cf", "ac", "afc", "sc", "sv", "club", "de", "del", "la", "le", "calcio", "1", "04", "05",
         "1899", "1900", "1909", "1846", "1907", "1910", "ssc", "us", "as", "rc", "rcd", "ud", "cd", "sd"}


def plain(text):
    """Lowercase, no accents, no punctuation: 'Club Atlético de Madrid' -> 'club atletico de madrid'."""
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9' ]", " ", text.lower())


def words(text, expand=False):
    found = []
    for word in plain(text).split():
        if expand and word in ABBREVIATIONS:
            found += ABBREVIATIONS[word].split()
        elif word not in NOISE:
            found.append(word.replace("'", ""))
    return set(found)


def similarity(our_name, team):
    """0..1: how well our short name fits a football-data.org team (full name, short name, 3-letter code)."""
    ours = words(our_name, expand=True)
    best = 0.0
    for candidate in (team["name"], team.get("shortName"), team.get("tla")):
        if not candidate:
            continue
        theirs = words(candidate)
        overlap = len(ours & theirs) / max(len(ours), 1) if theirs else 0.0
        text = difflib.SequenceMatcher(None, " ".join(sorted(ours)), " ".join(sorted(theirs))).ratio()
        best = max(best, 0.7 * overlap + 0.3 * text)
    return best


def pair_teams(our_names, teams):
    """Best one-to-one pairing of our names with football-data.org teams."""
    scores = np.array([[similarity(name, team) for team in teams] for name in our_names])
    rows, cols = linear_sum_assignment(-scores)
    return [(our_names[r], teams[c], scores[r, c]) for r, c in zip(rows, cols)]


def current_teams(matches, league):
    season = matches.loc[matches["league"] == league, "season"].max()
    games = matches[(matches["league"] == league) & (matches["season"] == season)]
    return sorted(set(games["home_team"]) | set(games["away_team"]))


def load_api_key():
    key = os.environ.get("FOOTBALL_DATA_API_KEY")
    env = Path(".env")
    if not key and env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("FOOTBALL_DATA_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"')
    if not key:
        raise SystemExit("No FOOTBALL_DATA_API_KEY in .env - get a free key at football-data.org")
    return key


def main():
    key = load_api_key()
    matches = pd.read_csv(MATCHES_FILE)
    rows = []
    for league, competition in COMPETITIONS.items():
        response = requests.get(API_URL.format(code=competition), headers={"X-Auth-Token": key}, timeout=30)
        response.raise_for_status()
        teams = response.json()["teams"]
        for our_name, team, score in pair_teams(current_teams(matches, league), teams):
            rows.append({"league": league, "team": our_name, "api_name": team["name"],
                         "crest_url": team["crest"], "similarity": round(score, 2)})
            flag = "   <- CHECK" if score < CHECK_BELOW else ""
            print(f"{league:4} {our_name:20} -> {team['name']:40} {score:.2f}{flag}")

    OUT_FILE.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_FILE, index=False)
    print(f"\nSaved {len(rows)} crests to {OUT_FILE}. Fix any wrong row marked CHECK by hand.")


if __name__ == "__main__":
    main()
