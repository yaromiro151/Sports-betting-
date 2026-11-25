"""Command line.

    python -m sportsedge run         # update data, grade old picks, write today's card (the daily job)
    python -m sportsedge update      # just pull data
    python -m sportsedge card        # card from data already pulled
    python -m sportsedge grade       # grade finished picks
    python -m sportsedge ratings nba # current team ratings
    python -m sportsedge backtest nfl --start 2024-09-01 --end 2026-02-15
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from . import backtest, card, config, store


def _card(leagues, days_ahead):
    sections = {}
    all_bets = []
    for lg in leagues:
        try:
            games, bets, _ = card.build(lg, days_ahead[lg])
        except ValueError as e:
            print(f"{lg}: {e}")
            continue
        sections[lg] = (games, bets)
        all_bets.append(bets)
    picks = card.record_picks(pd.concat(all_bets) if all_bets else pd.DataFrame())
    text = card.write_report(sections, picks)
    print(text)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sportsedge")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "update", "card", "grade"):
        p = sub.add_parser(name)
        p.add_argument("--leagues", default=",".join(config.SPORTS))
    r = sub.add_parser("ratings")
    r.add_argument("league", choices=list(config.LEAGUES))
    lv = sub.add_parser("live")
    lv.add_argument("--force", default="", help="team abbreviations to price even if the game is close")
    lv.add_argument("--min-edge", type=float, default=0.05)
    b = sub.add_parser("backtest")
    b.add_argument("league", choices=list(config.LEAGUES))
    b.add_argument("--start", required=True)
    b.add_argument("--end", required=True)
    args = ap.parse_args(argv)

    pd.set_option("display.width", 160)
    days_ahead = {"nba": 1, "nfl": 7}

    if args.cmd in ("run", "update", "card", "grade"):
        leagues = [s.strip() for s in args.leagues.split(",") if s.strip()]
        if args.cmd in ("run", "update"):
            print("update:", json.dumps(store.update(leagues)))
        if args.cmd in ("run", "grade"):
            card.grade_picks()
        if args.cmd in ("run", "card"):
            _card(leagues, days_ahead)
    elif args.cmd == "ratings":
        _, model = card.predict_upcoming(args.league, 0)
        print(model.ratings().round(2).to_string())
        print(f"home edge: {model.hfa:.2f} pts")
    elif args.cmd == "live":
        from . import live
        force = [t.strip().upper() for t in args.force.split(",") if t.strip()]
        games, priced, credits = live.check(force=force, min_edge=args.min_edge)
        if not games:
            print("No NBA games in progress.")
        for g in games:
            flag = "BET LIVE" if live.worth_betting(g) else "wait"
            print(f"{g['away']} {g['away_score']:.0f} @ {g['home']} {g['home_score']:.0f} | {g['detail']} | lead {g['lead']:.0f} | {flag}")
        if not priced.empty:
            good = priced[(priced.edge_over >= args.min_edge) & priced.rule10].sort_values("edge_over", ascending=False)
            print("\nLive overs (team in control, low rungs):")
            print(good[["game", "player", "stat", "line", "over", "now", "need", "mins_left", "model_over", "edge_over"]].to_string(index=False))
        if credits is not None:
            print(f"\nOdds API credits left: {credits}")
    elif args.cmd == "backtest":
        games = store.load_games(args.league)
        lines = store.load_nfl_closing_lines() if args.league == "nfl" else None
        res = backtest.run(args.league, games, args.start, args.end, lines)
        print("accuracy:", json.dumps(res.get("accuracy"), default=float))
        if "market" in res:
            print("vs closing line:", json.dumps(res["market"], default=float))
            print(res["bets"].to_string(index=False))


if __name__ == "__main__":
    main()
