"""Turn model predictions + sportsbook lines into priced bets."""
from __future__ import annotations

import math

import pandas as pd

from .pricing import expected_value, implied_prob, kelly, no_vig, outcome_probs


def _ok(x) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def price_game(g: dict, home_pts: float, away_pts: float, line: dict, params: dict,
               threshold: float) -> list[dict]:
    """Price every side of spread / total / moneyline for one game.

    `line['home_spread']` is the home team's spread (negative = home favored).
    """
    m, t = home_pts - away_pts, home_pts + away_pts
    sm, st = params["sigma_margin"], params["sigma_total"]
    L, T = line.get("home_spread"), line.get("total")
    w = params.get("model_weight", 1.0)
    # A huge gap between model and market usually means news the model can't see (injuries,
    # rest days, lineup changes). Flag the whole game rather than calling any side a bet.
    check = ((_ok(L) and abs(-m - L) > params["max_disagree_spread"]) or
             (_ok(T) and abs(t - T) > params["max_disagree_total"]))
    max_stake = params.get("max_stake", 0.02)
    bets = []

    def add(market, side, line_value, odds, other_odds, probs, market_probs):
        """probs: model (win, push, loss). market_probs: the same if the market line were exactly
        right, used as a fallback when the other side's price is missing."""
        if not _ok(odds):
            return
        win, push, loss = probs
        p_model = win / (win + loss) if win + loss else 0.0
        # Market's fair (no-vig) probability for this side, then blend in the model's view
        # (see config.model_weight). Blending in probability space keeps the model's
        # distribution assumptions from inventing edges on long-shot prices.
        if _ok(other_odds):
            p_market = no_vig(odds, other_odds)[0]
        else:
            mw, _, ml = market_probs
            p_market = mw / (mw + ml) if mw + ml else 0.5
        p = (1 - w) * p_market + w * p_model
        win, loss = p * (1 - push), (1 - p) * (1 - push)
        be = implied_prob(odds)
        edge = p - be
        status = "CHECK" if check else ("BET" if edge >= threshold else "")
        bets.append(dict(
            league=g["league"], game_id=g["game_id"], espn_id=g.get("espn_id"),
            game_date=g["game_date"], start_time=g["start_time"],
            matchup=f"{g['away']} @ {g['home']}", market=market, side=side,
            line=line_value, odds=odds, model_prob=round(p, 4), breakeven=round(be, 4),
            edge=round(edge, 4), ev=round(expected_value(win, loss, odds), 4),
            kelly_quarter=round(min(kelly(win, loss, odds) / 4, max_stake), 4), status=status,
        ))

    hso, aso = line.get("home_spread_odds"), line.get("away_spread_odds")
    oo, uo = line.get("over_odds"), line.get("under_odds")
    hml, aml = line.get("home_ml"), line.get("away_ml")
    if _ok(L):
        add("spread", f"{g['home']} {L:+g}", L, hso, aso, outcome_probs(m, sm, -L), outcome_probs(-L, sm, -L))
        add("spread", f"{g['away']} {-L:+g}", -L, aso, hso, outcome_probs(-m, sm, L), outcome_probs(L, sm, L))
    if _ok(T):
        add("total", f"Over {T:g}", T, oo, uo, outcome_probs(t, st, T), outcome_probs(T, st, T))
        add("total", f"Under {T:g}", T, uo, oo, outcome_probs(-t, st, -T), outcome_probs(-T, st, -T))
    if _ok(L):
        add("moneyline", f"{g['home']} ML", None, hml, aml,
            outcome_probs(m, sm, 0, can_push=False), outcome_probs(-L, sm, 0, can_push=False))
        add("moneyline", f"{g['away']} ML", None, aml, hml,
            outcome_probs(-m, sm, 0, can_push=False), outcome_probs(L, sm, 0, can_push=False))
    return bets


def grade(bet: dict, home_score: float, away_score: float) -> tuple[str, float]:
    """Return (W/L/P, profit per 1 unit) for a priced bet once the game is final."""
    from .pricing import decimal
    home, away = bet["matchup"].split(" @ ")[1], bet["matchup"].split(" @ ")[0]
    margin, total = home_score - away_score, home_score + away_score
    side, mkt = bet["side"], bet["market"]
    if mkt == "spread":
        x = (margin if side.startswith(home + " ") else -margin) + float(bet["line"])
    elif mkt == "total":
        x = (total - float(bet["line"])) * (1 if side.startswith("Over") else -1)
    else:
        x = margin if side.startswith(home + " ") else -margin
    if x > 0:
        return "W", round(decimal(float(bet["odds"])) - 1, 4)
    if x < 0:
        return "L", -1.0
    return "P", 0.0
