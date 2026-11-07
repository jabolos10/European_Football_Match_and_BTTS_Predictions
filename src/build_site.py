"""
Build the public website: upcoming predictions + an honest track record.

Steps:
  1) add the newest predictions (results/predictions_upcoming.csv) to the prediction log
     (results/predictions_log.csv). A prediction can still change before its match day,
     after that it is frozen - so we can never "fix" a prediction after the match,
  2) settle the log: match every logged prediction with the real result once it is played,
  3) write site/index.html with the upcoming matches and the track record.

Bookmaker odds are NOT shown on the site.
Run from the project root (after python -m src.predict_fixtures):  python -m src.build_site
"""
import hashlib
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.metrics import outcome_index, rps

PREDICTIONS_FILE = Path("results/predictions_upcoming.csv")
LOG_FILE = Path("results/predictions_log.csv")
MATCHES_FILE = Path("data/processed/domestic_matches.csv")
SITE_DIR = Path("site")
CRESTS_FILE = Path("data/crests.csv")  # made by src/fetch_crests.py (top 5 leagues only)
SHOW_CRESTS = True  # False -> every team gets the initials badge, no images from football-data.org
KEY = ["date", "home_team", "away_team"]
# league code -> (name, country, flag, colour used on the site)
LEAGUES = {
    "E0": ("Premier League", "England", "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "#a855f7"),
    "SP1": ("LaLiga", "Spain", "🇪🇸", "#ff5a4f"),
    "I1": ("Serie A", "Italy", "🇮🇹", "#3b82f6"),
    "D1": ("Bundesliga", "Germany", "🇩🇪", "#f43f5e"),
    "F1": ("Ligue 1", "France", "🇫🇷", "#22d3ee"),
    "POL": ("Ekstraklasa", "Poland", "🇵🇱", "#34d399"),
}
SOURCE_TZ = ZoneInfo("Europe/London")   # football-data.co.uk gives kick-off times in UK time
SITE_TZ = ZoneInfo("Europe/Warsaw")     # the site shows Central European time
LATEST_HOUR_UTC = 9  # on match day predictions may only be added/changed before 09:00 UTC (before the first kick-off)


def update_log(now):
    """Add new predictions to the log, without ever changing a prediction for a match that may have started."""
    new = pd.read_csv(PREDICTIONS_FILE)
    new["predicted_at"] = now.strftime("%Y-%m-%d %H:%M UTC")
    today = now.strftime("%Y-%m-%d")
    still_open = (new["date"] > today) | ((new["date"] == today) & (now.hour < LATEST_HOUR_UTC))
    new = new[still_open]

    if LOG_FILE.exists():
        log = pd.read_csv(LOG_FILE)
        replaced = log.set_index(KEY).index.isin(new.set_index(KEY).index)
        log = pd.concat([log[~replaced], new], ignore_index=True)
    else:
        log = new
    log = log.sort_values(KEY).reset_index(drop=True)
    log.to_csv(LOG_FILE, index=False)
    return log


def settle(log):
    """Attach real results to logged predictions of matches that have been played."""
    results = pd.read_csv(MATCHES_FILE, usecols=KEY + ["home_goals", "away_goals"])
    settled = log.merge(results, on=KEY, how="inner")
    settled["outcome"] = outcome_index(settled["home_goals"], settled["away_goals"])
    settled["btts"] = ((settled["home_goals"] > 0) & (settled["away_goals"] > 0)).astype(int)
    probs = settled[["home_win", "draw", "away_win"]].to_numpy()
    settled["pick_right"] = np.argmax(probs, axis=1) == settled["outcome"]
    return settled


def summary(settled):
    """Track record numbers for the site."""
    if settled.empty:
        return None
    probs = settled[["home_win", "draw", "away_win"]].to_numpy()
    btts_prob = settled["btts_goals_model"].to_numpy()
    strong = btts_prob >= 0.6
    return {
        "matches": len(settled),
        "accuracy": settled["pick_right"].mean(),
        "rps": rps(probs, settled["outcome"].to_numpy()),
        "btts_strong_n": int(strong.sum()),
        "btts_strong_hit": settled.loc[strong, "btts"].mean() if strong.any() else np.nan,
    }




# ---------- page ----------

PREFIXES = {"FC", "AC", "AS", "SC", "SV", "RB", "CF", "CD", "UD", "SD", "RC"}


def pct(value):
    return "–" if pd.isna(value) else f"{value:.0%}"


def load_crests():
    """{team: crest image URL} from data/crests.csv; empty if crests are off or the file is missing."""
    if not SHOW_CRESTS or not CRESTS_FILE.exists():
        return {}
    crests = pd.read_csv(CRESTS_FILE).dropna(subset=["crest_url"])
    return dict(zip(crests["team"], crests["crest_url"]))


CRESTS = load_crests()


def initials_badge(team, hidden=False):
    """A round badge with the club's initials and its own colour."""
    words = [w for w in team.replace("-", " ").split() if w.upper() not in PREFIXES] or [team]
    letters = (words[0][0] + words[1][0]) if len(words) > 1 else words[0][:3]
    hue = int(hashlib.md5(team.encode()).hexdigest(), 16) % 360
    return f'<span class="crest" style="--h:{hue}"{" hidden" if hidden else ""}>{escape(letters.upper())}</span>'


def badge(team):
    """The club crest (linked from football-data.org) or, without one, the initials badge.
    If the image fails to load in the browser, the initials badge is shown instead."""
    url = CRESTS.get(team)
    if not url:
        return initials_badge(team)
    return (f'<img class="crest-img" src="{escape(url)}" alt="" loading="lazy" '
            f'onerror="this.nextElementSibling.hidden=false;this.remove()">{initials_badge(team, hidden=True)}')


def local_kick_off(row):
    """Kick-off in Central European time (date and HH:MM); falls back to the date only."""
    try:
        uk = datetime.strptime(f"{row.date} {row.kick_off}", "%Y-%m-%d %H:%M").replace(tzinfo=SOURCE_TZ)
        local = uk.astimezone(SITE_TZ)
        return local.date(), local.strftime("%H:%M")
    except (TypeError, ValueError):
        return datetime.strptime(str(row.date), "%Y-%m-%d").date(), ""


def day_title(day):
    return f"{day:%A} <span>{day.day} {day:%B}</span>"


def prob_bar(home, draw, away):
    best = max(home, draw, away)
    parts = []
    for label, value, cls in [("1", home, "h"), ("X", draw, "d"), ("2", away, "a")]:
        parts.append(f'<span class="p {cls}{" fav" if value == best else ""}"><b>{label}</b>{pct(value)}</span>')
    return (f'<div class="bar"><i class="h" style="width:{home:.1%}"></i><i class="d" style="width:{draw:.1%}"></i>'
            f'<i class="a" style="width:{away:.1%}"></i></div><div class="plabels">{"".join(parts)}</div>')


def team_line(team, extra):
    return f'<div class="team">{badge(team)}<span class="name">{escape(team)}</span>{extra}</div>'


def match_card(g, time_text):
    strong = " strong" if g.btts_goals_model >= 0.6 else ""
    return (
        f'<article class="match"><div class="when">{time_text}</div>'
        f'<div class="teams">{team_line(g.home_team, f"<span class=xg>{g.xg_home:.1f}</span>")}'
        f'{team_line(g.away_team, f"<span class=xg>{g.xg_away:.1f}</span>")}</div>'
        f'<div class="probs">{prob_bar(g.home_win, g.draw, g.away_win)}</div>'
        f'<div class="btts{strong}"><small>BTTS</small><b>{pct(g.btts_goals_model)}</b></div></article>')


def result_card(g):
    pick = "ok" if g.pick_right else "bad"
    btts_right = (g.btts_goals_model >= 0.5) == bool(g.btts)
    score = lambda goals: f'<span class="score">{int(goals)}</span>'
    return (
        f'<article class="match result"><div class="when">{local_kick_off(g)[0]:%d %b}</div>'
        f'<div class="teams">{team_line(g.home_team, score(g.home_goals))}{team_line(g.away_team, score(g.away_goals))}</div>'
        f'<div class="probs">{prob_bar(g.home_win, g.draw, g.away_win)}</div>'
        f'<div class="checks"><span class="chip {pick}">1X2 {"✓" if g.pick_right else "✗"}</span>'
        f'<span class="chip {"ok" if btts_right else "bad"}">BTTS {"✓" if btts_right else "✗"}</span></div></article>')


def league_header(code, count, label):
    name, country, flag, _ = LEAGUES[code]
    return (f'<div class="league-head"><span class="flag">{flag}</span><div><h3>{name}</h3>'
            f'<small>{country}</small></div><span class="count">{count} {label}</span></div>')


def upcoming_html(predictions):
    sections = []
    for code in LEAGUES:
        games = predictions[predictions["league"] == code]
        if games.empty:
            continue
        by_day = {}
        for g in games.itertuples():
            day, time_text = local_kick_off(g)
            by_day.setdefault(day, []).append(match_card(g, time_text))
        days = "".join(f'<h4 class="day">{day_title(day)}</h4><div class="grid">{"".join(cards)}</div>'
                       for day, cards in sorted(by_day.items()))
        sections.append(f'<section class="league" data-league="{code}" style="--lg:{LEAGUES[code][3]}">'
                        f'{league_header(code, len(games), "matches")}{days}</section>')
    return "".join(sections) or '<p class="empty">No upcoming matches right now.</p>'


def record_html(settled, stats):
    if stats is None:
        return ('<div class="stats"><div class="stat wide"><b>Starting now</b>'
                '<span>The first predictions are logged - the track record appears after the next matchday.</span></div></div>')
    stats_html = (
        '<div class="stats">'
        f'<div class="stat"><b>{stats["matches"]}</b><span>matches settled</span></div>'
        f'<div class="stat"><b>{stats["accuracy"]:.0%}</b><span>favourite was right</span></div>'
        f'<div class="stat"><b>{stats["rps"]:.3f}</b><span>RPS · lower is better</span></div>'
        f'<div class="stat"><b>{pct(stats["btts_strong_hit"])}</b><span>BTTS hit when we said ≥ 60% '
        f'({stats["btts_strong_n"]})</span></div></div>')
    sections = []
    recent = settled.sort_values("date", ascending=False)
    for code in LEAGUES:
        games = recent[recent["league"] == code].head(12)
        if games.empty:
            continue
        cards = "".join(result_card(g) for g in games.itertuples())
        sections.append(f'<section class="league" data-league="{code}" style="--lg:{LEAGUES[code][3]}">'
                        f'{league_header(code, len(games), "latest")}<div class="grid">{cards}</div></section>')
    return stats_html + "".join(sections)


def filters_html(codes):
    buttons = ['<button class="pill active" data-filter="all">All leagues</button>']
    for code in codes:
        name, _, flag, color = LEAGUES[code]
        buttons.append(f'<button class="pill" data-filter="{code}" style="--lg:{color}">{flag} {name}</button>')
    return f'<nav class="filters">{"".join(buttons)}</nav>'


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Football Predictions</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700;800&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root {{ --bg:#0a0c10; --surface:#12161d; --surface2:#1a1f28; --line:#232a35; --fg:#eef1f5; --muted:#8b95a5;
  --accent:#c8f542; --home:#c8f542; --draw:#5b6577; --away:#4cc9f0; --ok:#3ddc84; --bad:#ff5d5d; }}
* {{ box-sizing:border-box; margin:0; }}
body {{ background:var(--bg); color:var(--fg); font:14px/1.45 Inter, system-ui, sans-serif; -webkit-font-smoothing:antialiased; }}
main {{ max-width:1180px; margin:0 auto; padding:0 16px 80px; }}
.top {{ display:flex; align-items:center; justify-content:space-between; gap:16px; padding:22px 0; border-bottom:1px solid var(--line); flex-wrap:wrap; }}
.brand {{ display:flex; align-items:center; gap:12px; }}
.logo {{ width:38px; height:38px; border-radius:10px; background:var(--accent); display:grid; place-items:center; color:#0a0c10;
  font:800 20px/1 'Barlow Condensed', sans-serif; }}
.brand h1 {{ font:800 28px/1 'Barlow Condensed', sans-serif; letter-spacing:.02em; text-transform:uppercase; }}
.brand p {{ color:var(--muted); font-size:13px; margin-top:4px; }}
.updated {{ font-size:12px; color:var(--muted); border:1px solid var(--line); border-radius:999px; padding:6px 12px; }}
.updated::before {{ content:""; display:inline-block; width:7px; height:7px; border-radius:50%; background:var(--ok); margin-right:8px; }}
.filters {{ display:flex; gap:8px; overflow-x:auto; padding:18px 0 4px; scrollbar-width:none; }}
.pill {{ flex:none; background:var(--surface); color:var(--fg); border:1px solid var(--line); border-radius:999px; padding:9px 16px;
  font:600 13px Inter, sans-serif; cursor:pointer; }}
.pill.active {{ border-color:var(--lg, var(--accent)); box-shadow:inset 0 0 0 1px var(--lg, var(--accent)); color:var(--lg, var(--accent)); }}
h2 {{ font:800 34px/1 'Barlow Condensed', sans-serif; text-transform:uppercase; letter-spacing:.02em; margin:42px 0 18px; }}
h2 small {{ display:block; font:500 13px Inter, sans-serif; color:var(--muted); text-transform:none; letter-spacing:0; margin-top:8px; }}
.stats {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(200px, 1fr)); gap:12px; margin-bottom:8px; }}
.stat {{ background:var(--surface); border:1px solid var(--line); border-radius:14px; padding:18px; }}
.stat b {{ display:block; font:800 38px/1 'Barlow Condensed', sans-serif; color:var(--accent); }}
.stat span {{ color:var(--muted); font-size:13px; }}
.stat.wide {{ grid-column:1/-1; }} .stat.wide b {{ font-size:26px; }}
.league {{ margin-top:22px; background:var(--surface); border:1px solid var(--line); border-radius:16px; padding:18px; position:relative; overflow:hidden; }}
.league::before {{ content:""; position:absolute; inset:0 0 auto 0; height:3px; background:var(--lg); }}
.league-head {{ display:flex; align-items:center; gap:12px; }}
.league-head .flag {{ font-size:26px; }}
.league-head h3 {{ font:800 24px/1 'Barlow Condensed', sans-serif; text-transform:uppercase; letter-spacing:.03em; color:var(--lg); }}
.league-head small {{ color:var(--muted); }}
.league-head .count {{ margin-left:auto; font-size:12px; color:var(--muted); background:var(--surface2); border-radius:999px; padding:5px 10px; }}
.day {{ font:700 15px/1 'Barlow Condensed', sans-serif; text-transform:uppercase; letter-spacing:.08em; margin:20px 0 10px; display:flex; align-items:center; gap:10px; }}
.day span {{ color:var(--muted); font-weight:600; }}
.day::after {{ content:""; flex:1; height:1px; background:var(--line); }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(500px, 1fr)); gap:10px; }}
.match {{ display:grid; grid-template-columns:48px minmax(0,1fr) 168px 58px; align-items:center; gap:14px;
  background:var(--surface2); border:1px solid transparent; border-radius:12px; padding:12px 14px; transition:border-color .15s; }}
.match:hover {{ border-color:var(--lg); }}
.when {{ font:700 15px 'Barlow Condensed', sans-serif; color:var(--muted); }}
.teams {{ display:grid; gap:6px; min-width:0; }}
.team {{ display:flex; align-items:center; gap:9px; min-width:0; }}
.crest {{ flex:none; width:26px; height:26px; border-radius:50%; display:grid; place-items:center; font:700 9.5px Inter, sans-serif;
  letter-spacing:.02em; color:#fff; background:linear-gradient(140deg, hsl(var(--h) 55% 45%), hsl(calc(var(--h) + 40) 60% 28%));
  box-shadow:inset 0 0 0 1.5px rgba(255,255,255,.18); }}
.crest[hidden] {{ display:none; }}
.crest-img {{ flex:none; width:26px; height:26px; object-fit:contain; }}
.name {{ font-weight:600; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.xg, .score {{ margin-left:auto; font-variant-numeric:tabular-nums; }}
.xg {{ color:var(--muted); font-size:12px; }} .xg::before {{ content:"xG "; opacity:.6; }}
.score {{ font:800 20px/1 'Barlow Condensed', sans-serif; }}
.bar {{ display:flex; height:6px; border-radius:99px; overflow:hidden; gap:2px; background:var(--line); }}
.bar i {{ display:block; }} .bar .h {{ background:var(--home); }} .bar .d {{ background:var(--draw); }} .bar .a {{ background:var(--away); }}
.plabels {{ display:flex; justify-content:space-between; margin-top:7px; font-size:12px; color:var(--muted); font-variant-numeric:tabular-nums; }}
.plabels b {{ margin-right:4px; font-weight:700; opacity:.7; }}
.p.fav {{ color:var(--fg); font-weight:700; }} .p.fav.h b {{ color:var(--home); opacity:1; }} .p.fav.a b {{ color:var(--away); opacity:1; }}
.btts {{ text-align:center; border-radius:10px; padding:6px 0; background:var(--bg); }}
.btts small {{ display:block; font-size:9.5px; letter-spacing:.08em; color:var(--muted); }}
.btts b {{ font:800 19px/1.1 'Barlow Condensed', sans-serif; }}
.btts.strong {{ background:var(--accent); color:#0a0c10; }} .btts.strong small {{ color:#0a0c10; opacity:.7; }}
.checks {{ display:grid; gap:5px; }}
.chip {{ font-size:11px; font-weight:700; text-align:center; border-radius:999px; padding:3px 0; }}
.chip.ok {{ background:rgba(61,220,132,.14); color:var(--ok); }} .chip.bad {{ background:rgba(255,93,93,.13); color:var(--bad); }}
.result .match, .match.result {{ grid-template-columns:48px minmax(0,1fr) 168px 64px; }}
.legend {{ display:flex; gap:16px; flex-wrap:wrap; color:var(--muted); font-size:12px; margin-top:-6px; }}
.legend i {{ display:inline-block; width:10px; height:10px; border-radius:3px; margin-right:6px; vertical-align:-1px; }}
.empty {{ color:var(--muted); }}
footer {{ margin-top:56px; padding-top:20px; border-top:1px solid var(--line); color:var(--muted); font-size:13px; }}
footer a {{ color:var(--accent); }}
.hidden {{ display:none; }}
@media (max-width:620px) {{
  .grid {{ grid-template-columns:1fr; }}
  .match {{ grid-template-columns:minmax(0,1fr) 58px; gap:10px 12px; }}
  .when {{ grid-column:1/-1; font-size:13px; }}
  .probs {{ grid-column:1/-1; order:3; }}
  .league {{ padding:14px; }}
  h2 {{ font-size:28px; }}
}}
</style></head>
<body><main>
<header class="top">
  <div class="brand"><div class="logo">FP</div><div><h1>Football Predictions</h1>
  <p>Machine-learning forecasts for Europe's top leagues and Ekstraklasa</p></div></div>
  <span class="updated">Updated {updated}</span>
</header>
{filters}
<h2>Upcoming matches<small>Kick-off times in Central European Time. Predictions are frozen on match day and settled after the final whistle.</small></h2>
<div class="legend"><span><i style="background:var(--home)"></i>1 home win</span><span><i style="background:var(--draw)"></i>X draw</span>
<span><i style="background:var(--away)"></i>2 away win</span><span><i style="background:var(--accent)"></i>BTTS ≥ 60%</span><span>xG = expected goals</span></div>
{upcoming}
<h2>Track record<small>Every prediction is logged before the match - nothing is changed afterwards.</small></h2>
{record}
<footer>Model: own Elo ratings with form, shots, home/away and head-to-head features · logistic regression for the result,
Poisson goals model for both teams to score · data from football-data.co.uk ·
<a href="https://github.com/jabolos10/European_Football_Match_and_BTTS_Predictions">code on GitHub</a></footer>
</main>
<script>
document.querySelectorAll('.pill').forEach(function (pill) {{
  pill.addEventListener('click', function () {{
    document.querySelectorAll('.pill').forEach(function (p) {{ p.classList.toggle('active', p === pill); }});
    var league = pill.dataset.filter;
    document.querySelectorAll('.league').forEach(function (section) {{
      section.classList.toggle('hidden', league !== 'all' && section.dataset.league !== league);
    }});
  }});
}});
</script>
</body></html>
"""


def main():
    now = datetime.now(timezone.utc)
    log = update_log(now)
    settled = settle(log)
    stats = summary(settled)
    predictions = pd.read_csv(PREDICTIONS_FILE)

    SITE_DIR.mkdir(exist_ok=True)
    html = PAGE.format(updated=now.astimezone(SITE_TZ).strftime("%d %b %Y, %H:%M"),
                       filters=filters_html([c for c in LEAGUES if c in set(predictions["league"])]),
                       upcoming=upcoming_html(predictions), record=record_html(settled, stats))
    (SITE_DIR / "index.html").write_text(html, encoding="utf-8")
    print(f"Log: {len(log)} predictions, {len(settled)} settled. Wrote {SITE_DIR / 'index.html'}")


if __name__ == "__main__":
    main()
