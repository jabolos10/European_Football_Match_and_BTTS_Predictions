"""
Pre-match features for every match.

Every feature is computed ONLY from matches played before the match it describes,
otherwise the model would be "peeking into the future" and the test would lie.

Feature groups (added one by one in src/evaluate_models.py to see what helps):
  elo          - Elo difference of the two teams
  league       - which league it is (home advantage and draw rates differ between leagues)
  form         - points, goals scored and goals conceded per match in the last 5 matches
  scoring      - in the last 10 matches: % with both teams scoring, % clean sheets, % failed to score
  shots        - shots on target made and allowed, shot accuracy and finishing in the last 10 matches
                 (top 5 leagues only - Ekstraklasa files have no shot statistics)
  venue        - the home team in its last 6 HOME matches, the away team in its last 6 AWAY matches:
                 goals scored, goals conceded, % BTTS, % clean sheets, % failed to score
  recent       - goals scored and conceded in the last 3 and last 10 matches (form uses 5),
                 so the model itself can decide how much the latest matches count
  h2h          - the last 6 meetings of these two teams: points of the home team, % BTTS, goals per match
  league_trend - % BTTS and goals per match in the league over its last 15 match days
  day          - day of the week (e.g. Saturdays in Ekstraklasa)
"""
import numpy as np
import pandas as pd

FORM_MATCHES = 5
SCORING_MATCHES = 10
SHOTS_MATCHES = 10
VENUE_MATCHES = 6
RECENT_WINDOWS = [3, 10]
H2H_MATCHES = 6
TREND_MATCH_DAYS = 15
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def feature_groups(leagues):
    """Names of the feature columns, grouped. The order is the order we add them in."""
    return {
        "elo": ["elo_diff"],
        "league": [f"league_{code}" for code in leagues],
        "form": ["home_ppg", "away_ppg", "home_goals_for", "away_goals_for",
                 "home_goals_against", "away_goals_against"],
        "scoring": ["home_btts_rate", "away_btts_rate", "home_clean_sheet_rate", "away_clean_sheet_rate",
                    "home_failed_to_score_rate", "away_failed_to_score_rate"],
        "shots": ["home_shots_target_for", "away_shots_target_for", "home_shots_target_against",
                  "away_shots_target_against", "home_shot_accuracy", "away_shot_accuracy",
                  "home_finishing", "away_finishing", "has_shot_stats"],
        "venue": ["home_venue_goals_for", "home_venue_goals_against", "home_venue_btts_rate",
                  "home_venue_clean_sheet_rate", "home_venue_failed_to_score_rate",
                  "away_venue_goals_for", "away_venue_goals_against", "away_venue_btts_rate",
                  "away_venue_clean_sheet_rate", "away_venue_failed_to_score_rate"],
        "recent": [f"{side}_goals_{kind}_{n}" for n in RECENT_WINDOWS
                   for side in ("home", "away") for kind in ("for", "against")],
        "h2h": ["h2h_home_ppg", "h2h_btts_rate", "h2h_goals", "h2h_count"],
        "league_trend": ["league_btts_trend", "league_goals_trend"],
        "day": [f"day_{day}" for day in DAYS],
    }


def team_view(matches):
    """Every match twice: once from the home team's point of view, once from the away team's."""
    def column(name):
        """Shot columns are missing in some tables (e.g. small test tables) - use NaN then."""
        return matches[name] if name in matches.columns else np.nan

    home = pd.DataFrame({
        "match_id": matches.index, "date": matches["date"], "season": column("season"),
        "team": matches["home_team"], "opponent": matches["away_team"], "is_home": 1,
        "goals_for": matches["home_goals"], "goals_against": matches["away_goals"],
        "shots_for": column("home_shots"), "shots_target_for": column("home_shots_target"),
        "shots_target_against": column("away_shots_target"),
    })
    away = pd.DataFrame({
        "match_id": matches.index, "date": matches["date"], "season": column("season"),
        "team": matches["away_team"], "opponent": matches["home_team"], "is_home": 0,
        "goals_for": matches["away_goals"], "goals_against": matches["home_goals"],
        "shots_for": column("away_shots"), "shots_target_for": column("away_shots_target"),
        "shots_target_against": column("home_shots_target"),
    })
    long = pd.concat([home, away], ignore_index=True)
    # Upcoming matches have no score yet: their results stay NaN, so they never count as 0 points / no BTTS
    played = long["goals_for"].notna() & long["goals_against"].notna()
    points = np.select(
        [long["goals_for"] > long["goals_against"], long["goals_for"] == long["goals_against"]], [3, 1], 0)
    long["points"] = np.where(played, points, np.nan)
    long["btts"] = np.where(played, (long["goals_for"] > 0) & (long["goals_against"] > 0), np.nan)
    long["clean_sheet"] = np.where(played, long["goals_against"] == 0, np.nan)
    long["failed_to_score"] = np.where(played, long["goals_for"] == 0, np.nan)
    long["total_goals"] = long["goals_for"] + long["goals_against"]
    # shot accuracy = shots on target / all shots, finishing = goals / shots on target
    long["shot_accuracy"] = long["shots_target_for"] / long["shots_for"].where(long["shots_for"] > 0)
    long["finishing"] = long["goals_for"] / long["shots_target_for"].where(long["shots_target_for"] > 0)
    return long.sort_values(["date", "match_id"], kind="stable").reset_index(drop=True)


def past_mean(long, by, column, window):
    """Average of `column` over the previous `window` rows of each group.

    shift(1) moves every value one row down, so the current match is NOT included.
    """
    return long.groupby(by)[column].transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())


def add_team_features(long):
    """Form, scoring habits and head-to-head record, for every team before every match."""
    long["ppg"] = past_mean(long, "team", "points", FORM_MATCHES)
    long["goals_for_avg"] = past_mean(long, "team", "goals_for", FORM_MATCHES)
    long["goals_against_avg"] = past_mean(long, "team", "goals_against", FORM_MATCHES)
    long["btts_rate"] = past_mean(long, "team", "btts", SCORING_MATCHES)
    long["clean_sheet_rate"] = past_mean(long, "team", "clean_sheet", SCORING_MATCHES)
    long["failed_to_score_rate"] = past_mean(long, "team", "failed_to_score", SCORING_MATCHES)

    for column in ["shots_target_for", "shots_target_against", "shot_accuracy", "finishing"]:
        long[f"{column}_avg"] = long.groupby("team")[column].transform(
            lambda s: s.shift(1).rolling(SHOTS_MATCHES, min_periods=1).mean())

    # Only home matches for the home team, only away matches for the away team
    venue = ["team", "is_home"]
    long["venue_goals_for"] = past_mean(long, venue, "goals_for", VENUE_MATCHES)
    long["venue_goals_against"] = past_mean(long, venue, "goals_against", VENUE_MATCHES)
    long["venue_btts_rate"] = past_mean(long, venue, "btts", VENUE_MATCHES)
    long["venue_clean_sheet_rate"] = past_mean(long, venue, "clean_sheet", VENUE_MATCHES)
    long["venue_failed_to_score_rate"] = past_mean(long, venue, "failed_to_score", VENUE_MATCHES)

    # Goals in the last 3 and last 10 matches, next to the last 5 from the form group
    for n in RECENT_WINDOWS:
        long[f"goals_for_{n}"] = past_mean(long, "team", "goals_for", n)
        long[f"goals_against_{n}"] = past_mean(long, "team", "goals_against", n)

    # how many matches the team has already played this season (to compare early vs late season)
    long["season_games"] = long.groupby(["team", "season"], dropna=False).cumcount()

    pair = ["team", "opponent"]  # the same two teams, from this team's point of view
    long["h2h_ppg"] = past_mean(long, pair, "points", H2H_MATCHES)
    long["h2h_btts"] = past_mean(long, pair, "btts", H2H_MATCHES)
    long["h2h_goals"] = past_mean(long, pair, "total_goals", H2H_MATCHES)
    long["h2h_count"] = long.groupby(pair).cumcount().clip(upper=H2H_MATCHES)  # meetings before this one
    return long


def league_trend(matches):
    """% BTTS and goals per match in each league over its previous match days (not the current day)."""
    played = matches["home_goals"].notna() & matches["away_goals"].notna()  # upcoming matches don't count
    df = pd.DataFrame({
        "league": matches["league"], "date": matches["date"], "played": played,
        "btts": ((matches["home_goals"] > 0) & (matches["away_goals"] > 0)).where(played),
        "goals": matches["home_goals"] + matches["away_goals"],
    })
    per_day = (df.groupby(["league", "date"])
                 .agg(btts=("btts", "sum"), goals=("goals", "sum"), n=("played", "sum"))
                 .reset_index()
                 .sort_values(["league", "date"]))

    def previous_days_sum(column):
        return per_day.groupby("league")[column].transform(
            lambda s: s.shift(1).rolling(TREND_MATCH_DAYS, min_periods=1).sum())

    n = previous_days_sum("n")
    per_day["league_btts_trend"] = previous_days_sum("btts") / n
    per_day["league_goals_trend"] = previous_days_sum("goals") / n
    return per_day[["league", "date", "league_btts_trend", "league_goals_trend"]]


def add_features(matches, leagues):
    """Add all feature columns to a table that already has elo_home and elo_away."""
    df = matches.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["elo_diff"] = (df["elo_home"] - df["elo_away"]) / 100  # scaled: 1.0 = 100 Elo points

    long = add_team_features(team_view(df))
    team_columns = ["ppg", "goals_for_avg", "goals_against_avg", "btts_rate",
                    "clean_sheet_rate", "failed_to_score_rate", "shots_target_for_avg",
                    "shots_target_against_avg", "shot_accuracy_avg", "finishing_avg", "season_games",
                    "venue_goals_for", "venue_goals_against", "venue_btts_rate",
                    "venue_clean_sheet_rate", "venue_failed_to_score_rate"]
    team_columns += [f"goals_{kind}_{n}" for n in RECENT_WINDOWS for kind in ("for", "against")]
    home = long[long["is_home"] == 1].set_index("match_id")
    away = long[long["is_home"] == 0].set_index("match_id")
    for column in team_columns:
        name = column.replace("_avg", "")
        df[f"home_{name}"] = home[column]
        df[f"away_{name}"] = away[column]
    # 1 = shot statistics exist for this league (Ekstraklasa files have none)
    df["has_shot_stats"] = (df["home_shots_target_for"].notna() | df["away_shots_target_for"].notna()).astype(int)
    df["h2h_home_ppg"] = home["h2h_ppg"]
    df["h2h_btts_rate"] = home["h2h_btts"]
    df["h2h_goals"] = home["h2h_goals"]
    df["h2h_count"] = home["h2h_count"]

    df = df.merge(league_trend(df), on=["league", "date"], how="left").set_index(df.index)

    for code in leagues:
        df[f"league_{code}"] = (df["league"] == code).astype(int)
    weekday = df["date"].dt.day_name()
    for day in DAYS:
        df[f"day_{day}"] = (weekday == day).astype(int)
    return df
