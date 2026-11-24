"""Storage.

- data/cache.db (SQLite, not committed): games and scores, rebuilt from the sources if lost.
- history/lines.csv  (committed): every DraftKings line snapshot we have taken.
- history/picks.csv  (committed): every flagged bet and, once graded, its result.
"""
from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta

import pandas as pd

from . import config
from .sources import espn, nflverse

GAME_COLS = ["league", "game_id", "espn_id", "season", "game_date", "start_time", "home", "away",
             "home_score", "away_score", "neutral", "status"]
LINE_COLS = ["league", "espn_id", "source", "book", "fetched_at", "home_spread", "home_spread_odds",
             "away_spread_odds", "total", "over_odds", "under_odds", "home_ml", "away_ml",
             "open_home_spread", "open_total"]


def today_et() -> date:
    return datetime.now(espn.ET).date()


def _db() -> sqlite3.Connection:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(config.DATA_DIR / "cache.db")
    con.execute("""CREATE TABLE IF NOT EXISTS games (
        league TEXT, game_id TEXT, espn_id TEXT, season INT, game_date TEXT, start_time TEXT,
        home TEXT, away TEXT, home_score REAL, away_score REAL, neutral INT, status TEXT,
        PRIMARY KEY (league, game_id))""")
    con.execute("CREATE TABLE IF NOT EXISTS fetched (league TEXT, day TEXT, PRIMARY KEY (league, day))")
    return con


def upsert_games(games: pd.DataFrame | list[dict]) -> None:
    df = pd.DataFrame(games, columns=GAME_COLS)
    if df.empty:
        return
    df = df.astype(object).where(df.notna(), None)
    with _db() as con:
        con.executemany(f"INSERT OR REPLACE INTO games VALUES ({','.join('?' * len(GAME_COLS))})",
                        df[GAME_COLS].itertuples(index=False, name=None))


def load_games(league: str) -> pd.DataFrame:
    with _db() as con:
        return pd.read_sql("SELECT * FROM games WHERE league = ? ORDER BY start_time", con, params=(league,))


def append_lines(lines: list[dict] | pd.DataFrame) -> None:
    df = pd.DataFrame(lines, columns=LINE_COLS)
    if df.empty:
        return
    path = config.HISTORY_DIR / "lines.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def load_lines() -> pd.DataFrame:
    path = config.HISTORY_DIR / "lines.csv"
    if not path.exists():
        return pd.DataFrame(columns=LINE_COLS)
    return pd.read_csv(path, dtype={"espn_id": str})


# ---------------------------------------------------------------- updating

def _fetch_day(league: str, day: date):
    return espn.parse_events(espn.scoreboard(league, day), league, config.BOOKMAKER)


def update_nba(lookahead_days: int = 2) -> dict:
    """Pull every NBA game day since NBA_HISTORY_START that we don't have yet, plus fresh lines."""
    today = today_et()
    start = date.fromisoformat(config.NBA_HISTORY_START)
    with _db() as con:
        done = {r[0] for r in con.execute("SELECT day FROM fetched WHERE league='nba'")}
    days = [start + timedelta(d) for d in range((today - start).days + lookahead_days + 1)]
    # Skip the off-season (July-September) and days already stored as complete.
    days = [d for d in days if d.month not in (7, 8, 9) and d.isoformat() not in done]
    games, lines, complete = [], [], []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for d, (g, l) in zip(days, pool.map(lambda d: _fetch_day("nba", d), days)):
            games += g
            lines += l
            if d < today - timedelta(days=1) and all(x["status"] == "final" for x in g):
                complete.append(d.isoformat())
    upsert_games(games)
    append_lines(lines)
    with _db() as con:
        con.executemany("INSERT OR IGNORE INTO fetched VALUES ('nba', ?)", [(d,) for d in complete])
    return dict(days_fetched=len(days), games=len(games), lines=len(lines))


def update_nfl() -> dict:
    """Reload nflverse schedules (scores + consensus closing lines) and DraftKings lines from ESPN."""
    today = today_et()
    current = today.year if today.month >= 3 else today.year - 1
    sched = nflverse.load_schedules(list(range(config.NFL_FIRST_SEASON, current + 1)))
    games = nflverse.to_games(sched)
    upsert_games(games)
    nflverse.to_closing_lines(sched).to_csv(config.DATA_DIR / "nfl_closing_lines.csv", index=False)
    days = [today + timedelta(d) for d in range(8)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        lines = [l for _, ls in pool.map(lambda d: _fetch_day("nfl", d), days) for l in ls]
    append_lines(lines)
    return dict(games=len(games), lines=len(lines))


def load_nfl_closing_lines() -> pd.DataFrame:
    path = config.DATA_DIR / "nfl_closing_lines.csv"
    return pd.read_csv(path, dtype={"espn_id": str}) if path.exists() else pd.DataFrame()


def update(leagues: list[str]) -> dict:
    out = {}
    for lg in leagues:
        out[lg] = update_nba() if lg == "nba" else update_nfl()
    return out
