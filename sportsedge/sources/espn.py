"""ESPN public scoreboard: NBA/NFL games, scores and DraftKings lines for upcoming games."""
from __future__ import annotations

import time
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import requests

BASE = "https://site.api.espn.com/apis/site/v2/sports"
PATH = {"nba": "basketball/nba", "nfl": "football/nfl"}
ET = ZoneInfo("America/New_York")

_session = requests.Session()
_session.headers["User-Agent"] = "sportsedge/0.1"


def get_json(url: str, params: dict | None = None, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            r = _session.get(url, params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def scoreboard(league: str, day: date | None = None, **params) -> dict:
    if day is not None:
        params["dates"] = day.strftime("%Y%m%d")
    params.setdefault("limit", 300)
    return get_json(f"{BASE}/{PATH[league]}/scoreboard", params)


def _num(value) -> float | None:
    """'-110' -> -110.0, 'o212.5' -> 212.5, 'EVEN' -> 100.0, junk -> None."""
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in ("even", "ev", "pk", "pick"):
        return 100.0 if s.startswith("ev") else 0.0
    s = s.lstrip("ou").replace("+", "")
    try:
        return float(s)
    except ValueError:
        return None


def _close(market: dict, side: str, key: str):
    return _num(((market or {}).get(side) or {}).get("close", {}).get(key))


def _open(market: dict, side: str, key: str):
    return _num(((market or {}).get(side) or {}).get("open", {}).get(key))


def parse_odds(comp: dict, book: str = "draftkings") -> dict | None:
    odds = comp.get("odds") or []
    if not odds:
        return None
    chosen = next((o for o in odds if book in (o.get("provider", {}).get("name", "").lower())), None)
    if chosen is None:
        return None
    o = chosen
    ps, tot, ml = o.get("pointSpread"), o.get("total"), o.get("moneyline")
    home_spread = _close(ps, "home", "line")
    if home_spread is None and o.get("spread") is not None:
        home_spread = float(o["spread"])  # ESPN's 'spread' field is the home team's line
    total = _close(tot, "over", "line")
    if total is None and o.get("overUnder") is not None:
        total = float(o["overUnder"])
    return dict(
        book=o.get("provider", {}).get("name", "").lower(),
        home_spread=home_spread,
        home_spread_odds=_close(ps, "home", "odds"),
        away_spread_odds=_close(ps, "away", "odds"),
        total=total,
        over_odds=_close(tot, "over", "odds"),
        under_odds=_close(tot, "under", "odds"),
        home_ml=_close(ml, "home", "odds"),
        away_ml=_close(ml, "away", "odds"),
        open_home_spread=_open(ps, "home", "line"),
        open_total=_open(tot, "over", "line"),
    )


def parse_events(data: dict, league: str, book: str = "draftkings") -> tuple[list[dict], list[dict]]:
    games, lines = [], []
    fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for e in data.get("events", []):
        season_type = (e.get("season") or {}).get("type")
        if season_type in (1, 4):  # preseason / off-season exhibitions
            continue
        comp = e["competitions"][0]
        sides = {c["homeAway"]: c for c in comp.get("competitors", [])}
        if set(sides) != {"home", "away"}:
            continue
        start = datetime.fromisoformat(e["date"].replace("Z", "+00:00"))
        status = comp["status"]["type"]
        final = bool(status.get("completed"))
        state = "final" if final else ("in_progress" if status.get("state") == "in" else "scheduled")
        g = dict(
            league=league,
            game_id=str(e["id"]),
            espn_id=str(e["id"]),
            season=(e.get("season") or {}).get("year"),
            game_date=start.astimezone(ET).date().isoformat(),
            start_time=start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            home=sides["home"]["team"]["abbreviation"],
            away=sides["away"]["team"]["abbreviation"],
            home_score=float(sides["home"]["score"]) if final else None,
            away_score=float(sides["away"]["score"]) if final else None,
            neutral=int(bool(comp.get("neutralSite"))),
            status=state,
        )
        games.append(g)
        if not final:
            odds = parse_odds(comp, book)
            if odds and odds["home_spread"] is not None:
                lines.append(dict(league=league, espn_id=g["espn_id"], source="espn",
                                  fetched_at=fetched_at, **odds))
    return games, lines
