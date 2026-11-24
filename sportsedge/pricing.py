"""Odds math: American odds, vig removal, win/push/loss probabilities, expected value, Kelly."""
from __future__ import annotations

import math

from scipy.stats import norm


def decimal(american: float) -> float:
    return 1 + (american / 100 if american > 0 else 100 / abs(american))


def implied_prob(american: float) -> float:
    return 1 / decimal(american)


def no_vig(a: float, b: float) -> tuple[float, float]:
    """Fair probabilities for a two-way market once the bookmaker's margin is removed."""
    pa, pb = implied_prob(a), implied_prob(b)
    return pa / (pa + pb), pb / (pa + pb)


def outcome_probs(mean: float, sd: float, threshold: float, can_push: bool = True) -> tuple[float, float, float]:
    """(win, push, loss) for a bet that wins when X > threshold, X ~ Normal(mean, sd).

    Scores are whole numbers, so on a whole-number line the half point either side counts as a push.
    """
    if can_push and float(threshold).is_integer():
        loss = norm.cdf((threshold - 0.5 - mean) / sd)
        win = 1 - norm.cdf((threshold + 0.5 - mean) / sd)
        return win, 1 - win - loss, loss
    win = 1 - norm.cdf((threshold - mean) / sd)
    return win, 0.0, 1 - win


def expected_value(win: float, loss: float, american: float) -> float:
    """Expected profit per 1 unit staked."""
    return win * (decimal(american) - 1) - loss


def kelly(win: float, loss: float, american: float) -> float:
    """Full-Kelly stake as a fraction of bankroll (0 if no edge). Pushes are ignored."""
    if win + loss <= 0:
        return 0.0
    p = win / (win + loss)
    b = decimal(american) - 1
    return max(0.0, (b * p - (1 - p)) / b)


def fmt_american(x: float | None) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return f"+{x:.0f}" if x > 0 else f"{x:.0f}"
