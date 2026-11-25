"""The Odds API: DraftKings NFL player props (main over/under lines + anytime TD)."""
from __future__ import annotations

import pandas as pd
import requests

from .oddsapi import _key

BASE = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl"
MARKETS = {"player_pass_yds": "pass_yds", "player_rush_yds": "rush_yds", "player_reception_yds": "rec_yds",
           "player_receptions": "rec", "player_anytime_td": "anytime_td"}


def events() -> list[dict]:
    r = requests.get(f"{BASE}/events", params={"apiKey": _key()}, timeout=30)
    r.raise_for_status()
    return r.json()


def props(event_ids: list[str], book: str = "draftkings") -> tuple[pd.DataFrame, int | None]:
    evs = {e["id"]: e for e in events()}
    rows, remaining = [], None
    for eid in event_ids:
        e = evs[eid]
        r = requests.get(f"{BASE}/events/{eid}/odds", timeout=30, params=dict(
            apiKey=_key(), regions="us", bookmakers=book, markets=",".join(MARKETS), oddsFormat="american"))
        r.raise_for_status()
        remaining = r.headers.get("x-requests-remaining", remaining)
        for b in r.json().get("bookmakers", []):
            for m in b["markets"]:
                sides = {}
                for o in m["outcomes"]:
                    sides.setdefault((o["description"], o.get("point")), {})[o["name"]] = o["price"]
                for (player, point), p in sides.items():
                    rows.append(dict(event_id=eid, commence=e["commence_time"], home=e["home_team"],
                                     away=e["away_team"], player=player, stat=MARKETS[m["key"]], line=point,
                                     over=p.get("Over", p.get("Yes")), under=p.get("Under", p.get("No"))))
    return pd.DataFrame(rows), (int(remaining) if remaining is not None else None)
