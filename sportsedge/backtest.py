"""Walk-forward backtest: refit before every game day using only earlier games, then predict."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .edges import grade, price_game
from .model import RatingModel, add_rest_flags


def walk_forward(games: pd.DataFrame, params: dict, start: str, end: str) -> pd.DataFrame:
    games = add_rest_flags(games, params["short_rest_days"])
    test = games[games.home_score.notna() & (games.game_date >= start) & (games.game_date <= end)]
    out = []
    model = RatingModel(params["half_life"], params["alpha"], params["min_team_games"])
    for day, day_games in test.groupby("game_date"):
        try:
            model.fit(games, day)
        except ValueError:
            continue
        hp, ap = model.predict(day_games)
        out.append(day_games.assign(pred_home=hp, pred_away=ap))
    if not out:
        return pd.DataFrame()
    df = pd.concat(out)
    # Drop exhibition games (e.g. All-Star) whose teams the model doesn't know.
    df = df[df.home.isin(model.teams_) & df.away.isin(model.teams_)]
    df["margin"] = df.home_score - df.away_score
    df["total"] = df.home_score + df.away_score
    df["pred_margin"] = df.pred_home - df.pred_away
    df["pred_total"] = df.pred_home + df.pred_away
    return df


def accuracy(df: pd.DataFrame) -> dict:
    err_m = df.pred_margin - df.margin
    err_t = df.pred_total - df.total
    return dict(
        games=len(df),
        margin_mae=round(err_m.abs().mean(), 2),
        margin_sd=round(err_m.std(), 2),
        total_mae=round(err_t.abs().mean(), 2),
        total_sd=round(err_t.std(), 2),
        winner_pct=round(((df.pred_margin > 0) == (df.margin > 0)).mean(), 3),
    )


def against_market(df: pd.DataFrame, lines: pd.DataFrame, params: dict,
                   thresholds=(0.0, 0.02, 0.03, 0.05)) -> tuple[dict, pd.DataFrame]:
    """Compare predictions with closing lines and simulate flat 1-unit bets on flagged sides."""
    lines = lines.drop(columns=["league", "espn_id"], errors="ignore").rename(columns={"total": "total_line"})
    j = df.merge(lines, on="game_id", how="inner")
    if j.empty:
        return {}, pd.DataFrame()
    market = dict(
        games_with_lines=len(j),
        market_margin_mae=round((-j.home_spread - j.margin).abs().mean(), 2),
        model_margin_mae=round((j.pred_margin - j.margin).abs().mean(), 2),
        market_total_mae=round((j.total_line - j.total).abs().mean(), 2),
        model_total_mae=round((j.pred_total - j.total).abs().mean(), 2),
    )

    bets = []
    for r in j.itertuples(index=False):
        g = r._asdict()
        line = dict(home_spread=g["home_spread"], home_spread_odds=g["home_spread_odds"],
                    away_spread_odds=g["away_spread_odds"], total=g["total_line"],
                    over_odds=g["over_odds"], under_odds=g["under_odds"],
                    home_ml=g["home_ml"], away_ml=g["away_ml"])
        for b in price_game(g, g["pred_home"], g["pred_away"], line, params, threshold=0.0):
            b["result"], b["profit"] = grade(b, g["home_score"], g["away_score"])
            bets.append(b)
    bets = pd.DataFrame(bets)
    rows = []
    for mkt in ["spread", "total", "moneyline"]:
        for th in thresholds:
            # One side per market per game at most: the side with the bigger edge.
            sel = bets[(bets.market == mkt) & (bets.edge >= th) & (bets.status != "CHECK")]
            sel = sel.sort_values("edge").groupby("game_id").tail(1)
            if sel.empty:
                continue
            decided = sel[sel.result != "P"]
            rows.append(dict(market=mkt, min_edge=th, bets=len(sel),
                             win_pct=round((decided.result == "W").mean(), 3),
                             roi=round(sel.profit.sum() / len(sel), 3)))
    return market, pd.DataFrame(rows)


def run(league: str, games: pd.DataFrame, start: str, end: str, lines: pd.DataFrame | None = None,
        params: dict | None = None) -> dict:
    params = dict(params or config.LEAGUES[league], max_stake=config.MAX_STAKE)
    df = walk_forward(games, params, start, end)
    if df.empty:
        return {"error": "no games in range"}
    out = {"accuracy": accuracy(df)}
    if lines is not None and not lines.empty:
        out["market"], out["bets"] = against_market(df, lines, params)
    return out
