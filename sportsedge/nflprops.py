"""NFL player props: nflverse weekly stats (+ ESPN box scores for weeks nflverse hasn't posted yet),
priced against DraftKings main lines.

Model per stat:
- yards (pass / rush / receiving): normal, mean = (0.6 season + 0.4 last-3) x opponent-allowed factor,
  sd blends the player's own game-to-game spread with a typical spread for that stat.
- receptions: negative binomial (variance = mu + 0.10 mu^2).
- anytime TD: Poisson on TDs per game, shrunk toward 0.25 with 3 games of prior.
Rules (NFL version of RULES.md): volume players only (starting QB, RB with 12+ carries, receivers
with 6+ targets), anytime TDs and unders are singles, no backups, questionable players are swaps,
no passing/receiving overs outdoors in 20+ mph wind, no favorite rushing overs in an upset spot,
and a main-line edge of 15%+ is treated as missing news, not a bet.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import nbinom, norm

from . import store
from .props import _norm
from .sources.espn import get_json

SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary"
INJURIES = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries"
STATS = ["pass_yds", "rush_yds", "rec_yds", "rec", "tds"]
CV = {"pass_yds": 0.28, "rush_yds": 0.45, "rec_yds": 0.55}

NFL_TEAMS = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR", "Chicago Bears": "CHI", "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL", "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC", "Los Angeles Rams": "LA", "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN", "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}
ESPN_TO_NFLVERSE = {"WSH": "WAS", "LAR": "LA"}


def weekly(season: int) -> pd.DataFrame:
    import nflreadpy as nfl
    s = nfl.load_player_stats([season]).to_pandas()
    s = s[s.season_type == "REG"]
    return pd.DataFrame(dict(
        player=s.player_display_name, pos=s.position, team=s.team, opp=s.opponent_team, week=s.week,
        attempts=s.attempts, pass_yds=s.passing_yards, carries=s.carries, rush_yds=s.rushing_yards,
        targets=s.targets, rec=s.receptions, rec_yds=s.receiving_yards,
        tds=s.rushing_tds.fillna(0) + s.receiving_tds.fillna(0))).fillna(0)


def _espn_box(espn_id: str, week: int) -> list[dict]:
    data = get_json(SUMMARY, {"event": espn_id})
    teams = [t["team"]["abbreviation"] for t in data["boxscore"]["players"]]
    rows: dict[tuple, dict] = {}
    for side in data["boxscore"]["players"]:
        team = ESPN_TO_NFLVERSE.get(side["team"]["abbreviation"], side["team"]["abbreviation"])
        opp = ESPN_TO_NFLVERSE.get(next(t for t in teams if t != side["team"]["abbreviation"]),
                                   next(t for t in teams if t != side["team"]["abbreviation"]))
        for cat in side["statistics"]:
            keys = cat.get("keys", [])
            for a in cat.get("athletes", []):
                v = dict(zip(keys, a.get("stats", [])))
                r = rows.setdefault((a["athlete"]["displayName"], team), dict(
                    player=a["athlete"]["displayName"], pos="", team=team, opp=opp, week=week, attempts=0,
                    pass_yds=0, carries=0, rush_yds=0, targets=0, rec=0, rec_yds=0, tds=0))
                f = lambda k: float(str(v.get(k, 0)).replace("--", "0") or 0)
                if cat["name"] == "passing":
                    r["pass_yds"] = f("passingYards")
                    ca = str(v.get("completions/passingAttempts", "0/0")).split("/")
                    r["attempts"] = float(ca[1]) if len(ca) == 2 else 0
                elif cat["name"] == "rushing":
                    r["carries"], r["rush_yds"] = f("rushingAttempts"), f("rushingYards")
                    r["tds"] += f("rushingTouchdowns")
                elif cat["name"] == "receiving":
                    r["rec"], r["rec_yds"], r["targets"] = f("receptions"), f("receivingYards"), f("receivingTargets")
                    r["tds"] += f("receivingTouchdowns")
    return list(rows.values())


def load(season: int = 2026) -> pd.DataFrame:
    """nflverse weeks plus any completed later weeks from ESPN box scores."""
    w = weekly(season)
    games = store.load_games("nfl")
    games = games[(games.season == season) & (games.status == "final") & games.espn_id.notna()]
    games = games.assign(week=games.game_id.str.split("_").str[1].astype(int))
    todo = games[games.week > w.week.max()]
    with ThreadPoolExecutor(max_workers=8) as pool:
        extra = [r for rows in pool.map(lambda g: _espn_box(g.espn_id, g.week), todo.itertuples()) for r in rows]
    if extra:
        e = pd.DataFrame(extra)
        pos = w.drop_duplicates("player", keep="last").set_index("player").pos
        e["pos"] = e.player.map(pos).fillna("")
        w = pd.concat([w, e], ignore_index=True)
    return w


def defense(w: pd.DataFrame) -> pd.DataFrame:
    """Per opponent: yards and catches allowed per game, as a factor of league average."""
    g = w.groupby(["opp", "week"])[["pass_yds", "rush_yds", "rec_yds", "rec", "tds"]].sum().reset_index()
    per = g.groupby("opp")[["pass_yds", "rush_yds", "rec_yds", "rec", "tds"]].mean()
    return per / per.mean()


def injuries() -> pd.DataFrame:
    data = get_json(INJURIES)
    rows = []
    for t in data.get("injuries", []):
        for i in t.get("injuries", []):
            rows.append(dict(team=t.get("displayName"), player=i.get("athlete", {}).get("displayName"),
                             status=i.get("status"), comment=i.get("shortComment")))
    return pd.DataFrame(rows)


def price(board: pd.DataFrame, w: pd.DataFrame, dfac: pd.DataFrame, inj: pd.DataFrame,
          upset_favorites: set[str], windy_homes: set[str] = frozenset()) -> pd.DataFrame:
    from .pricing import implied_prob
    w = w.assign(_n=w.player.map(_norm))
    status = {_norm(p): s for p, s in zip(inj.player, inj.status)} if not inj.empty else {}
    out = []
    for r in board.dropna(subset=["over"]).itertuples(index=False):
        n = _norm(r.player)
        g = w[w._n == n].sort_values("week")
        if len(g) < 3:
            continue
        st = status.get(n, "")
        if st in ("Out", "Doubtful", "Injured Reserve"):
            continue
        team = g.team.iloc[-1]
        home, away = NFL_TEAMS.get(r.home, r.home), NFL_TEAMS.get(r.away, r.away)
        opp = away if team == home else home
        stat = "tds" if r.stat == "anytime_td" else r.stat
        season, l3 = g[stat].mean(), g[stat].tail(3).mean()
        fac_key = {"pass_yds": "pass_yds", "rush_yds": "rush_yds", "rec_yds": "rec_yds", "rec": "rec", "tds": "tds"}[stat]
        fac = float(dfac.loc[opp, fac_key]) if opp in dfac.index else 1.0
        mu = (0.6 * season + 0.4 * l3) * fac
        if stat == "tds":
            lam = (len(g) * mu + 3 * 0.25) / (len(g) + 3)
            p_over = 1 - np.exp(-lam)
            line, hit = 0.5, (g.tds > 0)
        elif stat == "rec":
            r_ = 1 / 0.10
            p_over = float(nbinom.sf(np.floor(r.line), r_, r_ / (r_ + mu))) if mu > 0 else 0.0
            line, hit = r.line, g.rec > r.line
        else:
            sd = 0.5 * g[stat].std(ddof=0) + 0.5 * CV[stat] * mu
            p_over = float(1 - norm.cdf((r.line - mu) / max(sd, 1.0)))
            line, hit = r.line, g[stat] > r.line
        # NFL rules bank
        vol = {"pass_yds": g.attempts.mean() >= 25, "rush_yds": g.carries.mean() >= 12,
               "rec_yds": g.targets.mean() >= 6, "rec": g.targets.mean() >= 6, "tds": False}[stat]
        note = "" if vol else ("TD = single only" if stat == "tds" else "low volume")
        if vol and team in upset_favorites and stat == "rush_yds":
            vol, note = False, "upset check (favorite may trail and abandon the run)"
        if vol and home in windy_homes and stat in ("pass_yds", "rec_yds", "rec"):
            vol, note = False, "wind 20+ mph"
        if p_over - implied_prob(r.over) >= 0.15 and vol:
            vol, note = False, "edge too big: check role/news"
        if st == "Questionable":
            note = (note + "; " if note else "") + "QUESTIONABLE (swap)"
        imp_o = implied_prob(r.over)
        imp_u = implied_prob(r.under) if pd.notna(r.under) else np.nan
        out.append(dict(
            game=f"{away} @ {home}", kickoff=r.commence, player=r.player, team=team, opp=opp,
            stat=r.stat, line=line, over=r.over, under=r.under, mu=round(mu, 1), season=round(season, 1),
            l3=round(l3, 1), model_over=round(p_over, 3), edge_over=round(p_over - imp_o, 3),
            edge_under=round((1 - p_over) - imp_u, 3) if pd.notna(imp_u) else np.nan,
            hit_season=round(hit.mean(), 2), hit_l3=round(hit.tail(3).mean(), 2), games=len(g),
            rule_ok=bool(vol) and st != "Questionable", note=note,
            log=", ".join(str(int(x)) for x in g[stat].tail(5)),
        ))
    return pd.DataFrame(out)
