"""Daily card: predict upcoming games, price them against DraftKings, save picks, grade old picks."""
from __future__ import annotations

from datetime import timedelta

import pandas as pd
from scipy.stats import norm

from . import config, store
from .edges import grade, price_game
from .model import RatingModel, add_rest_flags
from .pricing import fmt_american

BOOK_NAMES = {"draftkings": "DraftKings"}

PICK_COLS = ["created_at", "league", "game_id", "espn_id", "game_date", "start_time", "matchup", "market",
             "side", "line", "odds", "model_prob", "breakeven", "edge", "ev", "kelly_quarter",
             "close_line", "close_odds", "clv", "result", "profit"]


def latest_lines(league: str) -> pd.DataFrame:
    """Most recent DraftKings snapshot per game."""
    lines = store.load_lines()
    lines = lines[(lines.league == league) & (lines.source == "espn")]
    return lines.sort_values("fetched_at").groupby("espn_id").tail(1)


def predict_upcoming(league: str, days_ahead: int) -> tuple[pd.DataFrame, RatingModel]:
    params = config.LEAGUES[league]
    games = add_rest_flags(store.load_games(league), params["short_rest_days"])
    today = store.today_et()
    model = RatingModel(params["half_life"], params["alpha"], params["min_team_games"]).fit(games, today)
    up = games[(games.status == "scheduled") & (games.game_date >= today.isoformat()) &
               (games.game_date <= (today + timedelta(days=days_ahead)).isoformat())].copy()
    if up.empty:
        return up, model
    up["pred_home"], up["pred_away"] = model.predict(up)
    return up, model


def build(league: str, days_ahead: int) -> tuple[pd.DataFrame, pd.DataFrame, RatingModel]:
    params = dict(config.LEAGUES[league], max_stake=config.MAX_STAKE)
    up, model = predict_upcoming(league, days_ahead)
    lines = latest_lines(league).set_index("espn_id")
    rows, bets = [], []
    for g in up.to_dict("records"):
        line = lines.loc[g["espn_id"]].to_dict() if g.get("espn_id") in lines.index else {}
        m, t = g["pred_home"] - g["pred_away"], g["pred_home"] + g["pred_away"]
        rows.append(dict(game_date=g["game_date"], start_time=g["start_time"],
                         matchup=f"{g['away']} @ {g['home']}",
                         model_score=f"{g['away']} {g['pred_away']:.0f} - {g['home']} {g['pred_home']:.0f}",
                         model_spread=round(-m, 1), dk_spread=line.get("home_spread"),
                         model_total=round(t, 1), dk_total=line.get("total"),
                         model_home_win=round(float(norm.cdf(m / params["sigma_margin"])), 3)))
        if line:
            bets += price_game(g, g["pred_home"], g["pred_away"], line, params, config.EDGE_THRESHOLD)
    return pd.DataFrame(rows), pd.DataFrame(bets), model


# ---------------------------------------------------------------- picks log

def load_picks() -> pd.DataFrame:
    path = config.HISTORY_DIR / "picks.csv"
    if not path.exists():
        return pd.DataFrame(columns=PICK_COLS)
    return pd.read_csv(path, dtype={"espn_id": str, "game_id": str, "result": str})


def save_picks(picks: pd.DataFrame) -> None:
    config.HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    picks[PICK_COLS].to_csv(config.HISTORY_DIR / "picks.csv", index=False)


def record_picks(bets: pd.DataFrame) -> pd.DataFrame:
    """Log BET-status sides. A side is logged once (first time it is flagged) and never re-priced,
    so the record reflects the price that was actually available when the card went out."""
    picks = load_picks()
    if bets.empty:
        return picks
    new = bets[bets.status == "BET"].copy()
    new["created_at"] = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    key = ["league", "game_id", "market", "side"]
    if not picks.empty:
        new["game_id"] = new.game_id.astype(str)
        seen = set(map(tuple, picks[key].astype(str).values))
        new = new[[tuple(map(str, r)) not in seen for r in new[key].values]]
    picks = pd.concat([picks, new.reindex(columns=PICK_COLS)], ignore_index=True)
    save_picks(picks)
    return picks


def _clv(p: dict, close: dict) -> tuple:
    """Closing line value: how much better our number was than the last line before the game."""
    home = p["matchup"].split(" @ ")[1]
    if p["market"] == "spread":
        is_home = p["side"].startswith(home + " ")
        cl = close["home_spread"] if is_home else -close["home_spread"]
        co = close["home_spread_odds"] if is_home else close["away_spread_odds"]
        return cl, co, round(float(p["line"]) - cl, 2)
    if p["market"] == "total":
        over = p["side"].startswith("Over")
        co = close["over_odds"] if over else close["under_odds"]
        diff = close["total"] - float(p["line"])
        return close["total"], co, round(diff if over else -diff, 2)
    from .pricing import implied_prob
    is_home = p["side"].startswith(home + " ")
    co = close["home_ml"] if is_home else close["away_ml"]
    if pd.isna(co):
        return None, None, None
    return None, co, round(implied_prob(co) - implied_prob(float(p["odds"])), 4)


def grade_picks() -> pd.DataFrame:
    picks = load_picks()
    if picks.empty:
        return picks
    lines = store.load_lines()
    lines = lines[lines.source == "espn"]
    for lg in picks.league.unique():
        games = store.load_games(lg).set_index("game_id")
        for i, p in picks[(picks.league == lg) & picks.result.isna()].iterrows():
            gid = str(p.game_id)
            if gid not in games.index or games.loc[gid, "status"] != "final":
                continue
            g = games.loc[gid]
            res, profit = grade(p.to_dict(), g.home_score, g.away_score)
            picks.loc[i, ["result", "profit"]] = [res, profit]
            pre = lines[(lines.espn_id == str(p.espn_id)) & (lines.fetched_at < g.start_time)]
            if not pre.empty:
                cl, co, clv = _clv(p.to_dict(), pre.sort_values("fetched_at").iloc[-1].to_dict())
                picks.loc[i, ["close_line", "close_odds", "clv"]] = [cl, co, clv]
    save_picks(picks)
    return picks


# ---------------------------------------------------------------- report

def _md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_none_\n"
    cols = list(df.columns)
    out = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    for r in df.itertuples(index=False):
        out += "| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |\n"
    return out


def record_summary(picks: pd.DataFrame) -> str:
    done = picks[picks.result.notna()]
    if done.empty:
        return "No graded picks yet. The record starts filling in as games finish.\n"
    w, l, p = (done.result == "W").sum(), (done.result == "L").sum(), (done.result == "P").sum()
    roi = done.profit.astype(float).sum() / len(done)
    clv = pd.to_numeric(done.clv, errors="coerce")
    s = f"**Record: {w}-{l}-{p}**, profit {done.profit.astype(float).sum():+.2f} units on {len(done)} bets (ROI {roi:+.1%}).\n"
    if clv.notna().any():
        s += (f"Beat the closing line on {(clv > 0).mean():.0%} of bets. Over the long run, that number "
              f"matters more than the win-loss record.\n")
    return s


def write_report(sections: dict[str, tuple[pd.DataFrame, pd.DataFrame]], picks: pd.DataFrame) -> str:
    today = store.today_et().isoformat()
    md = [f"# Daily card: {today}\n",
          f"Book: {BOOK_NAMES.get(config.BOOKMAKER, config.BOOKMAKER.title())} · "
          f"edge threshold {config.EDGE_THRESHOLD:.0%} · stake = % of bankroll "
          f"(quarter-Kelly, capped at {config.MAX_STAKE:.0%})\n",
          "## Track record\n", record_summary(picks)]
    for lg, (games, bets) in sections.items():
        md.append(f"## {lg.upper()}\n")
        if games.empty:
            md.append("No games in the window.\n")
            continue
        md.append("### Bets (model edge at or above threshold)\n")
        if not bets.empty:
            b = bets[bets.status == "BET"].sort_values("edge", ascending=False)
            b = b.assign(odds=b.odds.map(fmt_american), edge=(b.edge * 100).round(1).astype(str) + "%",
                         model_prob=(b.model_prob * 100).round(1).astype(str) + "%",
                         stake=(b.kelly_quarter * 100).round(2).astype(str) + "%")
            md.append(_md_table(b[["game_date", "matchup", "side", "odds", "model_prob", "edge", "stake"]]))
            lean = bets[(bets.status == "") & (bets.edge > 0)].sort_values("edge", ascending=False).head(5)
            if not lean.empty:
                lean = lean.assign(odds=lean.odds.map(fmt_american),
                                   edge=(lean.edge * 100).round(1).astype(str) + "%")
                md.append("\n**Leans** (positive but below threshold, not bets):\n")
                md.append(_md_table(lean[["game_date", "matchup", "side", "odds", "edge"]]))
            chk = bets[bets.status == "CHECK"][["matchup"]].drop_duplicates()
            if not chk.empty:
                md.append("\n**Check the news first** (model and market disagree a lot, usually because of an "
                          "injury or lineup change the model can't see): " + ", ".join(chk.matchup) + "\n")
        else:
            md.append("_No DraftKings lines available yet._\n")
        md.append("\n### All games\n")
        md.append(_md_table(games.drop(columns=["start_time"])))
    text = "\n".join(md)
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORT_DIR / f"{today}.md").write_text(text)
    (config.REPORT_DIR / "latest.md").write_text(text)
    return text
