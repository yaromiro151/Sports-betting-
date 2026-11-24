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
