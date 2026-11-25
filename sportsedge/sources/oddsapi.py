"""The Odds API: DraftKings NBA player props (main over/under lines).

The key is read from the ODDS_API_KEY environment variable (set it in the environment settings,
never in code). Each game x market costs one request credit; the remaining count is returned.
"""
from __future__ import annotations

import os

import pandas as pd
import requests

BASE = "https://api.the-odds-api.com/v4/sports/basketball_nba"
MARKETS = {"player_points": "pts", "player_rebounds": "reb", "player_assists": "ast", "player_threes": "3pm"}

TEAM_ABBR = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN", "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE", "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET", "Golden State Warriors": "GS", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM", "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NO",
    "New York Knicks": "NY", "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX", "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC", "San Antonio Spurs": "SA", "Toronto Raptors": "TOR", "Utah Jazz": "UTAH",
    "Washington Wizards": "WSH",
}


def _key() -> str:
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        raise RuntimeError("ODDS_API_KEY is not set (add it in the environment settings)")
    return key


def events() -> list[dict]:
    r = requests.get(f"{BASE}/events", params={"apiKey": _key()}, timeout=30)
    r.raise_for_status()
    return r.json()


def props(event_ids: list[str] | None = None, book: str = "draftkings") -> tuple[pd.DataFrame, int | None]:
    """One row per player x market: line, over price, under price. Returns (df, credits_remaining)."""
    evs = events()
    if event_ids:
        evs = [e for e in evs if e["id"] in event_ids]
    rows, remaining = [], None
    for e in evs:
        r = requests.get(f"{BASE}/events/{e['id']}/odds", timeout=30, params=dict(
            apiKey=_key(), regions="us", bookmakers=book, markets=",".join(MARKETS), oddsFormat="american"))
        r.raise_for_status()
        remaining = r.headers.get("x-requests-remaining", remaining)
        for b in r.json().get("bookmakers", []):
            for m in b["markets"]:
                sides = {}
                for o in m["outcomes"]:
                    sides.setdefault((o["description"], o.get("point")), {})[o["name"]] = o["price"]
                for (player, point), p in sides.items():
                    rows.append(dict(event_id=e["id"], commence=e["commence_time"],
                                     home=TEAM_ABBR.get(e["home_team"], e["home_team"]),
                                     away=TEAM_ABBR.get(e["away_team"], e["away_team"]),
                                     player=player, stat=MARKETS[m["key"]], line=point,
                                     over=p.get("Over"), under=p.get("Under")))
    return pd.DataFrame(rows), (int(remaining) if remaining is not None else None)
