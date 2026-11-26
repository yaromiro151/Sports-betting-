"""Live NFL props: stats so far (ESPN live box score) + what the rest of the game should add,
priced against DraftKings' live lines (Odds API).

Remaining output for each player = per-game rate x share of the game left, where the rate blends
his season average with tonight's pace (tonight counts more as the game goes on), then nudged by
game script: a team trailing by 10+ throws more and runs less; a team leading by 10+ does the
opposite. Yards use a normal distribution whose spread shrinks with the time left; counts
(catches, attempts, completions, carries, TDs, INTs) use Poisson / negative binomial.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import nbinom, norm, poisson

from .props import _norm
from .sources.espn import get_json
from .pricing import implied_prob

SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary"
ESPN_TO_NFLVERSE = {"WSH": "WAS", "LAR": "LA"}
STAT_MAP = {"pass_yds": "pass_yds", "rush_yds": "rush_yds", "rec_yds": "rec_yds", "rec": "rec",
            "anytime_td": "tds", "pass_att": "attempts", "pass_cmp": "completions", "pass_int": "ints",
            "pass_tds": "pass_tds", "rush_att": "carries"}
CV = {"pass_yds": 0.28, "rush_yds": 0.45, "rec_yds": 0.55}
PASSING = {"pass_yds", "attempts", "completions", "pass_tds", "rec_yds", "rec", "ints"}


def _f(v) -> float:
    try:
        return float(str(v).replace("--", "0"))
    except ValueError:
        return 0.0


def live_state(espn_id: str) -> tuple[dict, pd.DataFrame]:
    s = get_json(SUMMARY, {"event": espn_id})
    comp = s["header"]["competitions"][0]
    sides = {c["homeAway"]: c for c in comp["competitors"]}
    st = comp["status"]
    period = int(st.get("period", 0) or 0)
    clock = st.get("displayClock", "15:00")
    try:
        mm, ss = clock.split(":")
        left_q = int(mm) + int(ss) / 60
    except ValueError:
        left_q = 15.0
    played = 0.0 if period == 0 else min(60.0, (period - 1) * 15 + (15 - left_q))
    if st["type"].get("state") == "post":
        played = 60.0
    home = ESPN_TO_NFLVERSE.get(sides["home"]["team"]["abbreviation"], sides["home"]["team"]["abbreviation"])
    away = ESPN_TO_NFLVERSE.get(sides["away"]["team"]["abbreviation"], sides["away"]["team"]["abbreviation"])
    hs, as_ = _f(sides["home"].get("score", 0)), _f(sides["away"].get("score", 0))
    game = dict(home=home, away=away, home_score=hs, away_score=as_, played=played,
                detail=st["type"].get("shortDetail", ""))
    rows: dict[tuple, dict] = {}
    for side in s.get("boxscore", {}).get("players", []):
        team = ESPN_TO_NFLVERSE.get(side["team"]["abbreviation"], side["team"]["abbreviation"])
        for cat in side["statistics"]:
            keys = cat.get("keys", [])
            for a in cat.get("athletes", []):
                v = dict(zip(keys, a.get("stats", [])))
                r = rows.setdefault((a["athlete"]["displayName"], team), dict(
                    player=a["athlete"]["displayName"], team=team, attempts=0, completions=0, pass_yds=0,
                    pass_tds=0, ints=0, carries=0, rush_yds=0, rec=0, rec_yds=0, targets=0, tds=0))
                if cat["name"] == "passing":
                    ca = str(v.get("completions/passingAttempts", "0/0")).split("/")
                    r["completions"], r["attempts"] = (_f(ca[0]), _f(ca[1])) if len(ca) == 2 else (0, 0)
                    r["pass_yds"], r["pass_tds"], r["ints"] = _f(v.get("passingYards")), _f(v.get("passingTouchdowns")), _f(v.get("interceptions"))
                elif cat["name"] == "rushing":
                    r["carries"], r["rush_yds"] = _f(v.get("rushingAttempts")), _f(v.get("rushingYards"))
                    r["tds"] += _f(v.get("rushingTouchdowns"))
                elif cat["name"] == "receiving":
                    r["rec"], r["rec_yds"], r["targets"] = _f(v.get("receptions")), _f(v.get("receivingYards")), _f(v.get("receivingTargets"))
                    r["tds"] += _f(v.get("receivingTouchdowns"))
    return game, pd.DataFrame(rows.values())


def script_factor(team: str, game: dict, stat: str) -> float:
    margin = (game["home_score"] - game["away_score"]) * (1 if team == game["home"] else -1)
    late = game["played"] >= 30
    if abs(margin) < 10:
        return 1.0
    trailing = margin < 0
    k = 1.0 + (0.15 if late else 0.08) * (1 if trailing else -1)      # passing volume
    if stat in PASSING:
        return k
    if stat in ("rush_yds", "carries"):
        return 2 - k                                                    # rushing goes the other way
    return 1.0


def price(board: pd.DataFrame, now: pd.DataFrame, season: pd.DataFrame, game: dict) -> pd.DataFrame:
    season = season.assign(_n=season.player.map(_norm))
    now = now.assign(_n=now.player.map(_norm)).set_index("_n", drop=False)
    left = max(60.0 - game["played"], 0.0) / 60.0
    w_tonight = game["played"] / (game["played"] + 30.0)
    out = []
    for r in board.dropna(subset=["over"]).itertuples(index=False):
        stat = STAT_MAP.get(r.stat)
        n = _norm(r.player)
        hist = season[season._n == n]
        if stat is None or len(hist) < 3:
            continue
        cur = float(now.loc[n][stat]) if n in now.index else 0.0
        team = now.loc[n]["team"] if n in now.index else hist.team.iloc[-1]
        per_game = 0.6 * hist[stat].mean() + 0.4 * hist[stat].tail(3).mean()
        pace = cur / (game["played"] / 60.0) if game["played"] >= 5 else per_game
        rate = w_tonight * pace + (1 - w_tonight) * per_game
        exp_left = max(rate * left * script_factor(team, game, stat), 0.0)
        line = 0.5 if r.stat == "anytime_td" else float(r.line)
        need = np.floor(line) + 1 - cur
        if need <= 0:
            p = 1.0
        elif stat in CV:
            sd = max((0.5 * hist[stat].std(ddof=0) + 0.5 * CV[stat] * per_game) * np.sqrt(max(left, 0.05)), 3.0)
            p = float(1 - norm.cdf((need - 0.5 - exp_left) / sd))
        elif stat in ("rec", "attempts", "completions", "carries"):
            rr = 1 / 0.10
            p = float(nbinom.sf(need - 1, rr, rr / (rr + exp_left))) if exp_left > 0 else 0.0
        else:  # tds, pass_tds, ints
            p = float(poisson.sf(need - 1, exp_left))
        edge = p - implied_prob(r.over)
        out.append(dict(
            player=r.player, team=team, stat=r.stat, line=line, over=r.over, under=r.under, now=cur,
            need=max(need, 0), exp_left=round(exp_left, 1), model_over=round(p, 3), edge_over=round(edge, 3),
            edge_under=round((1 - p) - implied_prob(r.under), 3) if pd.notna(r.under) else np.nan,
            stale=bool(abs(edge) > 0.25 or need <= 0)))
    return pd.DataFrame(out)


def check(espn_id: str, odds_event_id: str, season_csv: str = "data/nfl_weekly_2026.csv"):
    """One live check of one NFL game: (game state, priced props, credits left)."""
    from .sources import oddsapi_nfl
    game, now = live_state(espn_id)
    board, credits = oddsapi_nfl.props([odds_event_id])
    season = pd.read_csv(season_csv)
    priced = price(board, now, season, game) if not board.empty else pd.DataFrame()
    return game, now, priced, credits
