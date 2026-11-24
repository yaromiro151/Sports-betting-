"""Score model: opponent-adjusted offense/defense ratings fit with time-weighted ridge regression.

For every completed game we write two rows, one per team on offense:

    points = league_avg + offense[team] + defense[opponent] + hfa * venue + rest effects

venue is +0.5 for the home team, -0.5 for the away team and 0 at a neutral site, so `hfa` is the
full home-court/field edge in points of margin. Recent games count more (exponential decay with
`half_life` days), and ridge shrinkage pulls teams with little data toward average.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


def add_rest_flags(games: pd.DataFrame, short_rest_days: int) -> pd.DataFrame:
    """Add home_short / away_short: 1 if the team played within `short_rest_days` days."""
    g = games.copy()
    g["_d"] = pd.to_datetime(g.game_date)
    long = pd.concat([
        g[["game_id", "_d", "home"]].rename(columns={"home": "team"}).assign(side="home"),
        g[["game_id", "_d", "away"]].rename(columns={"away": "team"}).assign(side="away"),
    ]).sort_values(["team", "_d"])
    long["rest"] = long.groupby("team")["_d"].diff().dt.days
    long["short"] = (long["rest"] <= short_rest_days).astype(int)
    piv = long.pivot_table(index="game_id", columns="side", values="short", aggfunc="first")
    g["home_short"] = g.game_id.map(piv.get("home")).fillna(0).astype(int)
    g["away_short"] = g.game_id.map(piv.get("away")).fillna(0).astype(int)
    return g.drop(columns="_d")


@dataclass
class RatingModel:
    half_life: float = 45.0
    alpha: float = 1.0
    min_team_games: int = 0
    teams_: list[str] = field(default_factory=list)
    coef_: np.ndarray | None = None
    intercept_: float = 0.0

    def _design(self, off, deff, venue, own_short, opp_short) -> np.ndarray:
        idx = {t: i for i, t in enumerate(self.teams_)}
        n = len(self.teams_)
        X = np.zeros((len(off), 2 * n + 3))
        for r, (o, d) in enumerate(zip(off, deff)):
            if o in idx:
                X[r, idx[o]] = 1.0
            if d in idx:
                X[r, n + idx[d]] = 1.0
        X[:, 2 * n] = venue
        X[:, 2 * n + 1] = own_short
        X[:, 2 * n + 2] = opp_short
        return X

    def fit(self, games: pd.DataFrame, as_of) -> "RatingModel":
        """Fit on completed games played strictly before `as_of`. `games` needs rest flags."""
        as_of = pd.Timestamp(as_of)
        dates = pd.to_datetime(games.game_date)
        g = games[games.home_score.notna() & (dates < as_of)].copy()
        age = (as_of - pd.to_datetime(g.game_date)).dt.days.to_numpy()
        w = 0.5 ** (age / self.half_life)
        keep = w > 0.005
        g, w = g[keep], w[keep]
        counts = pd.concat([g.home, g.away]).value_counts()
        ok = counts[counts >= self.min_team_games].index
        mask = (g.home.isin(ok) & g.away.isin(ok)).to_numpy()
        g, w = g[mask], w[mask]
        if len(g) < 20:
            raise ValueError("not enough completed games to fit")
        self.teams_ = sorted(set(g.home) | set(g.away))
        venue = np.where(g.neutral.to_numpy() == 1, 0.0, 0.5)
        X = np.vstack([
            self._design(g.home, g.away, venue, g.home_short, g.away_short),
            self._design(g.away, g.home, -venue, g.away_short, g.home_short),
        ])
        y = np.concatenate([g.home_score.to_numpy(), g.away_score.to_numpy()])
        ww = np.concatenate([w, w])
        m = Ridge(alpha=self.alpha).fit(X, y, sample_weight=ww)
        self.coef_, self.intercept_ = m.coef_, float(m.intercept_)
        return self

    def predict(self, games: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return predicted (home_points, away_points). Unknown teams get average ratings."""
        venue = np.where(games.neutral.to_numpy() == 1, 0.0, 0.5)
        hs = games.get("home_short", pd.Series(0, index=games.index)).to_numpy()
        as_ = games.get("away_short", pd.Series(0, index=games.index)).to_numpy()
        home = self._design(games.home, games.away, venue, hs, as_) @ self.coef_ + self.intercept_
        away = self._design(games.away, games.home, -venue, as_, hs) @ self.coef_ + self.intercept_
        return home, away

    @property
    def hfa(self) -> float:
        return float(self.coef_[2 * len(self.teams_)])

    def ratings(self) -> pd.DataFrame:
        n = len(self.teams_)
        off, deff = self.coef_[:n], self.coef_[n:2 * n]
        return (pd.DataFrame({"team": self.teams_, "offense": off, "defense": -deff, "net": off - deff})
                .sort_values("net", ascending=False).reset_index(drop=True))
