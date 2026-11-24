"""nflverse schedules: every NFL game since 1999 with scores and consensus closing lines."""
from __future__ import annotations

import pandas as pd


def load_schedules(seasons: list[int]) -> pd.DataFrame:
    import nflreadpy as nfl
    return nfl.load_schedules(seasons).to_pandas()


def _utc(day: pd.Series, clock: pd.Series) -> pd.Series:
    """nflverse kickoff times are US Eastern; store UTC like the ESPN data."""
    local = pd.to_datetime(day + " " + clock.fillna("13:00")).dt.tz_localize(
        "America/New_York", ambiguous="NaT", nonexistent="shift_forward")
    return local.dt.tz_convert("UTC").dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def to_games(s: pd.DataFrame) -> pd.DataFrame:
    s = s[s.game_type.isin(["REG", "WC", "DIV", "CON", "SB"])]
    final = s.home_score.notna() & s.away_score.notna()
    return pd.DataFrame(dict(
        league="nfl",
        game_id=s.game_id,
        espn_id=s.espn.astype("string"),
        season=s.season,
        game_date=s.gameday,
        start_time=_utc(s.gameday, s.gametime),
        home=s.home_team,
        away=s.away_team,
        home_score=s.home_score.where(final),
        away_score=s.away_score.where(final),
        neutral=(s.location == "Neutral").astype(int),
        status=final.map({True: "final", False: "scheduled"}),
    )).reset_index(drop=True)


def to_closing_lines(s: pd.DataFrame) -> pd.DataFrame:
    """Consensus closing lines in the same layout as ESPN lines (home_spread < 0 = home favored)."""
    s = s[s.spread_line.notna()]
    return pd.DataFrame(dict(
        league="nfl",
        game_id=s.game_id,
        espn_id=s.espn.astype("string"),
        source="nflverse_close",
        book="consensus",
        home_spread=-s.spread_line,
        home_spread_odds=s.home_spread_odds,
        away_spread_odds=s.away_spread_odds,
        total=s.total_line,
        over_odds=s.over_odds,
        under_odds=s.under_odds,
        home_ml=s.home_moneyline,
        away_ml=s.away_moneyline,
    )).reset_index(drop=True)
