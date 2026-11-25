"""NBA player props: verified ESPN box scores -> negative-binomial model on DraftKings milestone rungs.

Rules follow the prop-card playbook: high-volume shooters only for threes, center/star rebounds one
rung below, assists only for primary handlers, star points vs bad defenses, no bench/role players,
fade threes on back-to-backs, questionable players are swaps not legs.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import nbinom

from . import config, store
from .sources.espn import get_json

SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"
INJURIES = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
STATS = ["pts", "reb", "ast", "3pm"]
DISPERSION = {"pts": 0.05, "reb": 0.06, "ast": 0.09, "3pm": 0.18}   # variance = mu + k*mu^2
RUNGS = {"pts": [10, 15, 20, 25, 30, 35, 40], "reb": [4, 6, 8, 10, 12, 14],
         "ast": [2, 4, 6, 8, 10, 12], "3pm": [1, 2, 3, 4, 5, 6]}


def _split(s: str) -> tuple[int, int]:
    try:
        a, b = s.split("-")
        return int(a), int(b)
    except (ValueError, AttributeError):
        return 0, 0


def _parse_box(game: dict, data: dict) -> list[dict]:
    rows = []
    teams = [t["team"]["abbreviation"] for t in data["boxscore"]["players"]]
    for side in data["boxscore"]["players"]:
        team = side["team"]["abbreviation"]
        opp = next(t for t in teams if t != team)
        st = side["statistics"][0]
        keys = st["keys"]
        for a in st["athletes"]:
            if a.get("didNotPlay") or not a.get("stats"):
                continue
            v = dict(zip(keys, a["stats"]))
            fgm, fga = _split(v.get("fieldGoalsMade-fieldGoalsAttempted"))
            tpm, tpa = _split(v.get("threePointFieldGoalsMade-threePointFieldGoalsAttempted"))
            ftm, fta = _split(v.get("freeThrowsMade-freeThrowsAttempted"))
            num = lambda k: float(v.get(k) or 0) if str(v.get(k, "")).lstrip("-").isdigit() else 0.0
            rows.append(dict(
                game_id=game["game_id"], game_date=game["game_date"], team=team, opp=opp,
                home=int(team == game["home"]), player=a["athlete"]["displayName"],
                pid=str(a["athlete"]["id"]), starter=int(bool(a.get("starter"))),
                min=num("minutes"), pts=num("points"), reb=num("rebounds"), ast=num("assists"),
                **{"3pm": tpm}, tpa=tpa, fgm=fgm, fga=fga, ftm=ftm, fta=fta,
                oreb=num("offensiveRebounds"), tov=num("turnovers"),
            ))
    return rows


def load_boxscores(season: int) -> pd.DataFrame:
    """Every player line of the given ESPN season (e.g. 2027 = 2026-27), cached incrementally."""
    path = config.DATA_DIR / f"nba_box_{season}.csv"
    have = pd.read_csv(path, dtype={"game_id": str, "pid": str}) if path.exists() else pd.DataFrame()
    games = store.load_games("nba")
    games = games[(games.season == season) & (games.status == "final")]
    todo = games[~games.game_id.isin(set(have.get("game_id", [])))].to_dict("records")

    def fetch(g):
        try:
            return _parse_box(g, get_json(SUMMARY, {"event": g["espn_id"]}))
        except Exception:
            return []
    with ThreadPoolExecutor(max_workers=8) as pool:
        new = [r for rows in pool.map(fetch, todo) for r in rows]
    box = pd.concat([have, pd.DataFrame(new)], ignore_index=True)
    if new:
        box.to_csv(path, index=False)
    return box.sort_values(["game_date", "game_id"]).reset_index(drop=True)


def team_defense(box: pd.DataFrame) -> pd.DataFrame:
    """Per team: what opponents average against it (pts/reb/ast/3pm, 3P%, OREB%) and pace."""
    tg = box.groupby(["game_id", "team", "opp"])[["pts", "reb", "ast", "3pm", "tpa", "fga", "fgm",
                                                   "fta", "oreb", "tov"]].sum().reset_index()
    tg["poss"] = tg.fga - tg.oreb + tg.tov + 0.44 * tg.fta
    allowed = tg.rename(columns={"team": "offense", "opp": "team"})
    own = tg[["game_id", "team", "fga", "fgm", "oreb", "poss"]].rename(
        columns={"fga": "own_fga", "fgm": "own_fgm", "oreb": "own_oreb", "poss": "own_poss"})
    j = allowed.merge(own, on=["game_id", "team"])
    agg = j.groupby("team").agg(
        games=("game_id", "nunique"), pts=("pts", "mean"), reb=("reb", "mean"), ast=("ast", "mean"),
        **{"3pm": ("3pm", "mean")}, tpm_sum=("3pm", "sum"), tpa_sum=("tpa", "sum"),
        opp_oreb=("oreb", "sum"), own_miss=("own_fga", "sum"), own_made=("own_fgm", "sum"),
        poss=("poss", "mean"), own_poss=("own_poss", "mean"))
    agg["opp_3p_pct"] = agg.tpm_sum / agg.tpa_sum
    # Opponent offensive-rebound rate: share of this team's... misses? No: of the OPPONENT's misses,
    # approximated with opponent OREB / (opponent FGA - FGM) from the allowed rows.
    miss = j.groupby("team").apply(lambda d: (d.fga - d.fgm).sum(), include_groups=False)
    agg["opp_oreb_rate"] = agg.opp_oreb / miss
    agg["pace"] = (agg.poss + agg.own_poss) / 2
    for s in STATS:
        agg[f"{s}_rank"] = agg[s].rank(ascending=True).astype(int)   # 1 = allows the fewest
    return agg.drop(columns=["tpm_sum", "tpa_sum", "opp_oreb", "own_miss", "own_made", "poss", "own_poss"])


def load_injuries() -> pd.DataFrame:
    data = get_json(INJURIES)
    rows = []
    for t in data.get("injuries", []):
        for i in t.get("injuries", []):
            a = i.get("athlete", {})
            rows.append(dict(team_name=t.get("displayName"), player=a.get("displayName"),
                             status=i.get("status"), comment=i.get("shortComment"), date=i.get("date")))
    return pd.DataFrame(rows)


def prob_at_least(mu: float, k: float, rung: int) -> float:
    """P(X >= rung) for a negative binomial with mean mu and variance mu + k*mu^2."""
    if mu <= 0:
        return 0.0
    r = 1 / k
    p = r / (r + mu)
    return float(nbinom.sf(rung - 1, r, p))


def player_model(box: pd.DataFrame, defense: pd.DataFrame, team: str, opp: str, b2b: bool,
                 min_minutes: float = 28.0) -> pd.DataFrame:
    """Model every regular starter of `team` against `opp` on each stat's milestone rungs."""
    league = defense[STATS].mean()
    tb = box[box.team == team]
    out = []
    for pid, g in tb.groupby("pid"):
        g = g.sort_values("game_date")
        if len(g) < 5 or g["min"].tail(10).mean() < min_minutes:
            continue
        last5, last3 = g.tail(5), g.tail(3)
        for s in STATS:
            season, l5, l3 = g[s].mean(), last5[s].mean(), last3[s].mean()
            mu = (0.6 * season + 0.4 * l5) * (defense.loc[opp, s] / league[s]) * (0.96 if b2b else 1.0)
            tag = ""
            if l3 > season * 1.15 and l3 - season >= 1:
                tag = "RISING"
            elif l3 < season / 1.15 and season - l3 >= 1:
                tag = "FALLING"
            for rung in RUNGS[s]:
                p = prob_at_least(mu, DISPERSION[s], rung)
                if p < 0.5 or p > 0.97:
                    continue
                out.append(dict(
                    player=g.player.iloc[-1], pid=pid, team=team, opp=opp, stat=s, rung=rung,
                    mu=round(mu, 1), season_avg=round(season, 1), l5_avg=round(l5, 1),
                    model_p=round(p, 3),
                    hit_season=round((g[s] >= rung).mean(), 2),
                    hit_l10=round((g[s].tail(10) >= rung).mean(), 2),
                    hit_l5=round((last5[s] >= rung).mean(), 2),
                    games=len(g), minutes=round(g["min"].tail(10).mean(), 1),
                    tpa=round(g.tpa.mean(), 1), starter_share=round(g.starter.mean(), 2),
                    opp_rank=int(defense.loc[opp, f"{s}_rank"]), tag=tag,
                    log=", ".join(str(int(x)) for x in g[s].tail(10)),
                ))
    return pd.DataFrame(out)


def apply_rules(cands: pd.DataFrame, box: pd.DataFrame, b2b_teams: set[str]) -> pd.DataFrame:
    """Keep only legs the playbook allows; pick one rung per player-stat."""
    if cands.empty:
        return cands
    c = cands[cands.starter_share >= 0.8].copy()
    # primary handler = team's assist leader among regulars
    team_ast_lead = (box.groupby(["team", "pid"]).ast.mean().reset_index()
                     .sort_values("ast").groupby("team").tail(1).set_index("team").pid)
    keep = []
    for r in c.itertuples(index=False):
        ok = False
        if r.stat == "3pm":
            ok = r.tpa >= 7 and r.team not in b2b_teams
        elif r.stat == "reb":
            ok = r.season_avg >= 8
        elif r.stat == "ast":
            ok = team_ast_lead.get(r.team) == r.pid and r.season_avg >= 5
        elif r.stat == "pts":
            ok = r.season_avg >= 20 and r.opp_rank >= 13      # star vs a below-average defense
        keep.append(ok)
    c = c[keep]
    if c.empty:
        return c
    # Rebounds: one rung below the highest rung the model clears; others: highest rung >= 60%.
    picks = []
    for (pid, s), g in c.groupby(["pid", "stat"]):
        g = g.sort_values("rung")
        good = g[g.model_p >= 0.6]
        if good.empty:
            continue
        if s == "reb" and len(good) > 1:
            picks.append(good.iloc[-2])
        else:
            picks.append(good.iloc[-1])
    return pd.DataFrame(picks).sort_values("model_p", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------- pricing a DraftKings board

def _norm(name: str) -> str:
    import re
    n = re.sub(r"[.'’]", "", str(name)).lower()
    n = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", n)
    return " ".join(n.split())


def price_board(board: pd.DataFrame, box: pd.DataFrame, defense: pd.DataFrame,
                out_players: set[str], b2b_teams: set[str], upset_favorites: set[str]) -> pd.DataFrame:
    """Price every over/under on a DraftKings main-line board and tag it with the rules bank.

    rule_ok marks overs that pass RULES.md (center/star rebounds one rung under the average,
    high-volume threes, primary-handler assists, star points vs a below-average defense, no
    favorite-star overs in an upset spot). Unders are priced too but are singles only (rule 11).
    """
    from .pricing import implied_prob
    league = defense[STATS].mean()
    box = box.assign(_n=box.player.map(_norm))
    ast_lead = (box.groupby(["team", "_n"]).ast.mean().reset_index()
                .sort_values("ast").groupby("team").tail(1).set_index("team")._n)
    out_n = {_norm(p) for p in out_players}
    rows = []
    for r in board.dropna(subset=["line", "over"]).itertuples(index=False):
        n = _norm(r.player)
        g = box[box._n == n].sort_values("game_date")
        if g.empty or n in out_n:
            continue
        team = g.team.iloc[-1]
        opp = r.away if team == r.home else r.home
        s = r.stat
        season, l5, l3 = g[s].mean(), g[s].tail(5).mean(), g[s].tail(3).mean()
        mu = (0.6 * season + 0.4 * l5) * (defense.loc[opp, s] / league[s]) * (0.96 if team in b2b_teams else 1)
        rung = int(np.floor(r.line)) + 1                       # over 5.5 == 6+
        p_over = prob_at_least(mu, DISPERSION[s], rung)
        tag = ("RISING" if l3 > season * 1.15 and l3 - season >= 1 else
               "FALLING" if l3 < season / 1.15 and season - l3 >= 1 else "")
        minutes = g["min"].tail(10).mean()
        starter = g.starter.mean()
        ok, why = False, ""
        if starter < 0.8 or minutes < 26:
            why = "role player"
        elif s == "reb":
            ok = season >= 8 and rung <= season - 1 + 0.5
            why = "" if ok else ("wing/guard reb" if season < 8 else "line above avg-1")
        elif s == "3pm":
            ok = g.tpa.mean() >= 7 and team not in b2b_teams
            why = "" if ok else "low-volume threes"
        elif s == "ast":
            ok = ast_lead.get(team) == n and season >= 5
            if ok and defense.loc[opp, "ast_rank"] <= 5:
                ok, why = False, "top-5 assist D: drop a rung"
            elif not ok:
                why = "not primary handler"
        elif s == "pts":
            ok = season >= 20 and defense.loc[opp, "pts_rank"] >= 13
            why = "" if ok else ("not a star" if season < 20 else "good defense")
        if ok and team in upset_favorites and season >= 18:
            ok, why = False, "upset check"
        imp_o = implied_prob(r.over)
        imp_u = implied_prob(r.under) if pd.notna(r.under) else np.nan
        rows.append(dict(
            matchup=f"{r.away} @ {r.home}", player=r.player, team=team, opp=opp, stat=s, line=r.line,
            over=r.over, under=r.under, mu=round(mu, 1), season=round(season, 1), l5=round(l5, 1),
            model_over=round(p_over, 3), edge_over=round(p_over - imp_o, 3),
            edge_under=round((1 - p_over) - imp_u, 3) if pd.notna(imp_u) else np.nan,
            hit_season=round((g[s] >= rung).mean(), 2), hit_l10=round((g[s].tail(10) >= rung).mean(), 2),
            hit_l5=round((g[s].tail(5) >= rung).mean(), 2), tag=tag, rule_ok=ok, rule_note=why,
            opp_rank=int(defense.loc[opp, f"{s}_rank"]), log=", ".join(str(int(x)) for x in g[s].tail(5)),
        ))
    return pd.DataFrame(rows)
