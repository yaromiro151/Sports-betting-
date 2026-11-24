# sportsedge

Predicts NBA and NFL scores and compares them with DraftKings prices. Every day it writes a
**daily card** (`reports/latest.md`) listing each game, the model's score, spread and total next
to DraftKings' numbers, and any bet where the model's edge clears the threshold. Every flagged
bet is logged and graded automatically, so you can see whether the model actually beats the
book over time.

## Where it stands (read this first)

Tested walk-forward on past seasons, with the model only ever seeing games played before the one
it predicts:

| | NFL 2024-26 (733 games) | NBA 2025-27 (1,500 games) |
|---|---|---|
| Average miss on final margin, model | 10.2 pts | 11.5 pts |
| Average miss on final margin, DraftKings/consensus closing line | **9.6 pts** | no history yet |
| Picks the right winner | 63% | 68% |

**The closing line is still more accurate than the model.** That's normal for a first version
built only on scores. Sportsbook main lines are very sharp, and most published "edge" models
don't beat them. So the system is built to be honest about it:

- It **blends** the model with the market's own fair price (`model_weight` in
  `sportsedge/config.py`: NBA 25%, NFL 15%). It only calls a bet when the model's view is strong
  enough to clear a 3% edge.
- If the model and DraftKings disagree by a huge amount, the game is marked **Check the news**
  instead of a bet. That gap almost always means an injury or lineup change the model can't see.
- Stakes are quarter-Kelly, capped at 2% of bankroll.
- Every pick records its **closing line value (CLV)**: did the line move in your direction after
  the pick? A positive average CLV over a few hundred bets is the real sign of an edge.

Expect few or no bets on most days until the model improves. That's the system working, not
broken.

## How it runs by itself

`.github/workflows/daily-card.yml` runs everything at 10am and 6pm Eastern every day. It pulls
data, grades yesterday's picks, writes the new card, and commits `reports/` and `history/` back
to the repo. GitHub only runs scheduled workflows from the **default branch**, so it starts once
this code is merged into `main`. You can also trigger it by hand from the repo's **Actions** tab
("daily-card", then "Run workflow").

## Running it yourself

```bash
pip install -r requirements.txt
python -m sportsedge run                      # the daily job: update, grade, card
python -m sportsedge ratings nba              # current team power ratings
python -m sportsedge backtest nfl --start 2024-09-01 --end 2026-02-15
python -m pytest -q
```

## Where the data comes from (no API keys needed)

| What | Source |
|---|---|
| NBA games and scores | ESPN public scoreboard |
| NFL games, scores, historical closing lines | [nflverse](https://github.com/nflverse) |
| DraftKings spread / total / moneyline for upcoming games | ESPN scoreboard (DraftKings feed) |

The NBA's own stats site (`stats.nba.com`) blocks cloud servers, so it isn't used. An Odds API
key (`ODDS_API_KEY`) isn't needed for the current version. It's only useful later, for player
props or comparing several sportsbooks.

## Files

- `sportsedge/model.py`: team offense/defense ratings (time-weighted ridge regression, home edge, back-to-backs)
- `sportsedge/edges.py`: prices every spread / total / moneyline side and grades results
- `sportsedge/backtest.py`: walk-forward testing against real closing lines
- `sportsedge/card.py`: daily card, pick log, grading, CLV
- `history/lines.csv`: every DraftKings line snapshot taken (this builds the NBA line history over time)
- `history/picks.csv`: every flagged bet and its result
- `reports/`: one card per day; `latest.md` is always the newest

## Settings (environment variables)

`SPORTS` (default `nba,nfl`), `EDGE_THRESHOLD` (default `0.03`), `MAX_STAKE` (default `0.02`),
`BOOKMAKER` (default `draftkings`).

## Best next improvements, in order of likely value

1. **Injuries and lineups.** Use ESPN's injury report to adjust for missing stars. This is the
   biggest blind spot right now, especially in the NBA.
2. **NFL quarterback adjustment and play-by-play efficiency** (EPA per play from nflverse),
   which predicts better than points.
3. **Player props.** Books price these less sharply than main lines. That needs the Odds API key.
4. Once a few months of picks are logged, raise `model_weight` only if the CLV record supports it.

Bet responsibly. This is a research tool, not a guarantee.
