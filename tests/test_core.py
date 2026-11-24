import numpy as np
import pandas as pd
import pytest

from sportsedge import pricing
from sportsedge.edges import grade, price_game
from sportsedge.model import RatingModel, add_rest_flags
from sportsedge.sources.espn import parse_odds


def test_american_odds():
    assert pricing.decimal(-110) == pytest.approx(1.9091, abs=1e-4)
    assert pricing.decimal(150) == pytest.approx(2.5)
    a, b = pricing.no_vig(-110, -110)
    assert a == pytest.approx(0.5) and b == pytest.approx(0.5)


def test_outcome_probs_push_and_symmetry():
    win, push, loss = pricing.outcome_probs(0, 13, 3)        # whole-number line can push
    assert push > 0 and win + push + loss == pytest.approx(1)
    win, push, loss = pricing.outcome_probs(0, 13, 2.5)      # half-point line cannot
    assert push == 0
    w1, _, _ = pricing.outcome_probs(3, 13, 0, can_push=False)
    w2, _, _ = pricing.outcome_probs(-3, 13, 0, can_push=False)
    assert w1 + w2 == pytest.approx(1)


def test_kelly_zero_without_edge():
    assert pricing.kelly(0.5, 0.5, -110) == 0
    assert pricing.kelly(0.6, 0.4, -110) > 0


def test_grade_spread_total_ml():
    bet = dict(matchup="AAA @ BBB", market="spread", side="BBB -3.5", line=-3.5, odds=-110)
    assert grade(bet, 110, 105)[0] == "W"
    assert grade(bet, 106, 105)[0] == "L"
    bet = dict(matchup="AAA @ BBB", market="spread", side="AAA +3", line=3, odds=-110)
    assert grade(bet, 103, 100)[0] == "P"
    bet = dict(matchup="AAA @ BBB", market="total", side="Under 200.5", line=200.5, odds=-110)
    assert grade(bet, 100, 99)[0] == "W"
    bet = dict(matchup="AAA @ BBB", market="moneyline", side="AAA ML", line=None, odds=150)
    assert grade(bet, 99, 100) == ("W", 1.5)


def _synthetic(n_days=150, seed=0):
    rng = np.random.default_rng(seed)
    teams = [f"T{i:02d}" for i in range(12)]
    strength = dict(zip(teams, np.linspace(-6, 6, 12)))
    rows, gid = [], 0
    start = pd.Timestamp("2025-01-01")
    for d in range(n_days):
        order = rng.permutation(teams)
        for h, a in zip(order[::2], order[1::2]):
            mu = 110 + (strength[h] - strength[a]) / 2
            rows.append(dict(league="nba", game_id=str(gid), espn_id=str(gid), season=2025,
                             game_date=(start + pd.Timedelta(days=d)).date().isoformat(),
                             start_time="", home=h, away=a, neutral=0, status="final",
                             home_score=round(mu + 1.5 + rng.normal(0, 10)),
                             away_score=round(110 - (strength[h] - strength[a]) / 2 - 1.5 + rng.normal(0, 10))))
            gid += 1
    return pd.DataFrame(rows), strength


def test_model_recovers_strength_and_home_edge():
    games, strength = _synthetic()
    games = add_rest_flags(games, 1)
    m = RatingModel(half_life=1000, alpha=1.0).fit(games, "2026-01-01")
    r = m.ratings().set_index("team").net
    assert np.corrcoef([strength[t] for t in r.index], r.values)[0, 1] > 0.95
    assert m.hfa == pytest.approx(3.0, abs=1.0)


def test_price_game_blend_and_check():
    g = dict(league="nba", game_id="1", espn_id="1", game_date="2026-01-01", start_time="",
             home="BBB", away="AAA")
    line = dict(home_spread=-3.5, home_spread_odds=-110, away_spread_odds=-110, total=220.5,
                over_odds=-110, under_odds=-110, home_ml=-160, away_ml=135)
    params = dict(sigma_margin=14, sigma_total=19, max_disagree_spread=7, max_disagree_total=10,
                  model_weight=0.0)
    bets = price_game(g, 112, 108, line, params, 0.03)
    # model_weight 0 -> pure market: no side should be called a bet
    assert len(bets) == 6 and not any(b["status"] == "BET" for b in bets)
    bets = price_game(g, 130, 100, line, dict(params, model_weight=1.0), 0.03)
    assert all(b["status"] == "CHECK" for b in bets)   # 30-point gap vs a 3.5 line = check the news


def test_parse_espn_odds():
    comp = {"odds": [{"provider": {"name": "DraftKings"}, "spread": 2.5, "overUnder": 212.5,
                      "pointSpread": {"home": {"close": {"line": "+2.5", "odds": "-117"}},
                                      "away": {"close": {"line": "-2.5", "odds": "-104"}}},
                      "total": {"over": {"close": {"line": "o212.5", "odds": "-113"}},
                                "under": {"close": {"line": "u212.5", "odds": "EVEN"}}},
                      "moneyline": {"home": {"close": {"odds": "+110"}}, "away": {"close": {"odds": "-130"}}}}]}
    o = parse_odds(comp)
    assert o["home_spread"] == 2.5 and o["total"] == 212.5
    assert o["under_odds"] == 100.0 and o["home_ml"] == 110 and o["away_spread_odds"] == -104
