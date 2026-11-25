"""Live NBA props: what has happened so far + what the rest of the game should add, priced against
DraftKings' live lines (rule 10 in RULES.md: live role-player legs for the team in control).

Flow (one call = one check):
1. ESPN scoreboard (free): find games in progress and how lopsided they are.
2. ESPN live box score (free) for each live game: every player's minutes and stats so far.
3. Odds API live props (4 credits per game) only for games worth betting live, i.e. second half
   with a 10+ point lead (the script is clear), or any game you name.
4. For each prop: needed = line + 0.5 - current. Remaining output ~ negative binomial with mean =
   per-minute rate (blend of tonight and season) x expected minutes left. Starters lose minutes in
   blowouts (20+ lead in the 4th: half their remaining minutes).
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import nbinom

from . import props as P
from .sources import espn
from .sources.espn import get_json

DISP = P.DISPERSION
GAME_MIN = 48.0


def game_clock(comp: dict) -> tuple[int, float]:
    st = comp["status"]
    period = int(st.get("period", 0))
    clock = float(st.get("clock", 0.0)) / 60.0          # minutes left in the period
    played = min(period - 1, 3) * 12 + (12 - clock) if period <= 4 else 48 + (period - 5) * 5 + (5 - clock)
    return period, max(played, 0.0)


def live_games() -> list[dict]:
    data = espn.scoreboard("nba")
    out = []
    for e in data.get("events", []):
        c = e["competitions"][0]
        if c["status"]["type"].get("state") != "in":
            continue
        sides = {x["homeAway"]: x for x in c["competitors"]}
        period, played = game_clock(c)
        hs, as_ = float(sides["home"]["score"]), float(sides["away"]["score"])
        out.append(dict(espn_id=e["id"], home=sides["home"]["team"]["abbreviation"],
                        away=sides["away"]["team"]["abbreviation"], home_score=hs, away_score=as_,
                        period=period, played=round(played, 1), lead=abs(hs - as_),
                        leader=sides["home"]["team"]["abbreviation"] if hs > as_ else sides["away"]["team"]["abbreviation"],
                        detail=c["status"]["type"].get("shortDetail", "")))
    return out


def worth_betting(g: dict) -> bool:
    """Rule 10: the script is clear. Second half and a 10+ point lead."""
    return g["played"] >= 24 and g["lead"] >= 10


def live_box(espn_id: str, game: dict) -> pd.DataFrame:
    data = get_json(P.SUMMARY, {"event": espn_id})
    g = dict(game_id=espn_id, game_date="", home=game["home"])
    return pd.DataFrame(P._parse_box(g, data))


def project(row: pd.Series, season: pd.DataFrame, game: dict, stat: str) -> tuple[float, float]:
    """(expected remaining output, expected remaining minutes) for one player."""
    hist = season[season._n == row._n]
    # recent role matters more than the season average (rotations change)
    season_min = (0.5 * hist["min"].mean() + 0.5 * hist.sort_values("game_date")["min"].tail(5).mean()
                  if len(hist) else max(row["min"], 1.0) * GAME_MIN / max(game["played"], 1))
    season_rate = (hist[stat].sum() / hist["min"].sum()) if len(hist) and hist["min"].sum() > 0 else 0.0
    tonight_rate = row[stat] / row["min"] if row["min"] > 0 else season_rate
    # trust tonight more as minutes pile up (half weight at 20 minutes played)
    w = row["min"] / (row["min"] + 20.0)
    rate = w * tonight_rate + (1 - w) * season_rate
    left_game = max(GAME_MIN - game["played"], 0.0)
    # tonight's minutes pace counts when a player is clearly in a bigger role than usual
    pace_min = row["min"] / max(game["played"], 1.0) * GAME_MIN * 0.9 if game["played"] >= 12 else 0.0
    mins_left = min(max(max(season_min, pace_min) - row["min"], 0.0), left_game)
    if game["played"] >= 36 and game["lead"] >= 20:      # garbage time is coming
        if row.get("starter", 0) == 1:
            mins_left *= 0.5 if game["played"] >= 40 else 0.8                      # starters sit late
        else:
            mins_left = min(mins_left * 1.4 + 2.0, left_game)                      # bench plays more
    if row.get("pf", 0) >= 5:
        mins_left *= 0.6                                  # foul trouble
    return rate * mins_left, mins_left


def price_live(board: pd.DataFrame, box_now: pd.DataFrame, season_box: pd.DataFrame, game: dict) -> pd.DataFrame:
    from .pricing import implied_prob
    season_box = season_box.assign(_n=season_box.player.map(P._norm))
    box_now = box_now.assign(_n=box_now.player.map(P._norm)).set_index("_n", drop=False)
    rows = []
    for r in board.dropna(subset=["line", "over"]).itertuples(index=False):
        n = P._norm(r.player)
        if n not in box_now.index:
            continue
        b = box_now.loc[n]
        if isinstance(b, pd.DataFrame):
            b = b.iloc[0]
        cur = float(b[r.stat])
        exp_left, mins_left = project(b, season_box, game, r.stat)
        need = int(np.floor(r.line)) + 1 - cur            # more needed to cash the over
        if need <= 0:
            p_over = 1.0
        elif exp_left <= 0:
            p_over = 0.0
        else:
            k = DISP[r.stat]
            rr = 1 / k
            p_over = float(nbinom.sf(need - 1, rr, rr / (rr + exp_left)))
        hist = season_box[season_box._n == n]
        is_role = (hist.starter.mean() < 0.8) if len(hist) else True
        lead_team = b["team"] == game["leader"]
        rows.append(dict(
            game=f"{game['away']} @ {game['home']} ({game['detail']}, {game['away_score']:.0f}-{game['home_score']:.0f})",
            player=r.player, team=b["team"], stat=r.stat, line=r.line, over=r.over, under=r.under,
            now=cur, mins=round(float(b["min"]), 1), mins_left=round(mins_left, 1), need=max(need, 0),
            exp_left=round(exp_left, 1), model_over=round(p_over, 3),
            edge_over=round(p_over - implied_prob(r.over), 3),
            edge_under=round((1 - p_over) - implied_prob(r.under), 3) if pd.notna(r.under) else np.nan,
            # rule 10: role players on the team in control, low rungs only
            rule10=bool(lead_team and game["lead"] >= 10 and need <= max(2, 0.6 * exp_left)),
            role=is_role,
        ))
    return pd.DataFrame(rows)


def check(event_map: dict[str, str] | None = None, force: list[str] | None = None,
          min_edge: float = 0.05) -> tuple[list[dict], pd.DataFrame, int | None]:
    """One live check. event_map: ESPN id -> Odds API event id (built automatically if omitted)."""
    from .sources import oddsapi
    games = live_games()
    import os
    if not os.environ.get("ODDS_API_KEY"):
        print("ODDS_API_KEY not set: showing live scores only (add the key in the environment settings).")
        return games, pd.DataFrame(), None
    season_box = P.load_boxscores(2027)
    if event_map is None:
        evs = oddsapi.events()
        lookup = {(oddsapi.TEAM_ABBR.get(e["away_team"]), oddsapi.TEAM_ABBR.get(e["home_team"])): e["id"] for e in evs}
        event_map = {g["espn_id"]: lookup.get((g["away"], g["home"])) for g in games}
    out, remaining = [], None
    for g in games:
        if not (worth_betting(g) or (force and (g["home"] in force or g["away"] in force))):
            continue
        eid = event_map.get(g["espn_id"])
        if not eid:
            continue
        board, remaining = oddsapi.props([eid])
        if board.empty:
            continue
        out.append(price_live(board, live_box(g["espn_id"], g), season_box, g))
    priced = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    return games, priced, remaining
