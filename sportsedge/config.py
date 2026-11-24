"""Settings. Everything can be overridden with environment variables."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("SPORTSEDGE_DATA", ROOT / "data"))   # rebuildable cache (not committed)
HISTORY_DIR = ROOT / "history"                                       # line snapshots + picks (committed)
REPORT_DIR = ROOT / "reports"                                        # daily cards (committed)

SPORTS = [s.strip().lower() for s in os.environ.get("SPORTS", "nba,nfl").split(",") if s.strip()]
BOOKMAKER = os.environ.get("BOOKMAKER", "draftkings").lower()
EDGE_THRESHOLD = float(os.environ.get("EDGE_THRESHOLD", "0.03"))
ODDS_API_KEY = os.environ.get("ODDS_API_KEY")

# First date of history to pull for the NBA (ESPN) and first season for the NFL (nflverse).
NBA_HISTORY_START = os.environ.get("NBA_HISTORY_START", "2024-10-01")
NFL_FIRST_SEASON = int(os.environ.get("NFL_FIRST_SEASON", "2022"))

# Model settings per league. sigma_* are the typical errors of the model's margin / total,
# measured by `python -m sportsedge backtest` (see README). max_disagree: if the model's
# line differs from the market by more than this many points, the game is marked CHECK
# (usually injury/lineup news the model can't see) instead of being called a bet.
# model_weight: how much the model's own number counts vs the market line when pricing a bet
# (0 = trust the market fully, 1 = trust the model fully). On 2023-2026 NFL closing lines the
# best weight was ~0, i.e. the model adds nothing over the market yet; raise it only when the
# backtest / live closing-line record justifies it.
LEAGUES = {
    "nba": dict(half_life=40.0, alpha=3.0, sigma_margin=14.5, sigma_total=18.9,
                short_rest_days=1, max_disagree_spread=7.0, max_disagree_total=10.0,
                min_team_games=10, model_weight=0.25),
    "nfl": dict(half_life=200.0, alpha=5.0, sigma_margin=13.0, sigma_total=13.6,
                short_rest_days=5, max_disagree_spread=6.0, max_disagree_total=7.0,
                min_team_games=4, model_weight=0.15),
}
MAX_STAKE = float(os.environ.get("MAX_STAKE", "0.02"))  # never suggest more than 2% of bankroll on one bet
