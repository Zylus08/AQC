"""
aqc/research/tournament/alpha_leaderboard.py
==============================================
Leaderboard generation for Alpha Tournament.

Formats the raw AlphaRanker output into production-ready data structures
for dashboards and reports.

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging

import pandas as pd

from aqc.alpha.alpha_ranker import AlphaRanker

logger = logging.getLogger(__name__)


class AlphaLeaderboard:
    """Formats and manages the tournament leaderboard."""

    def __init__(self, ranker: AlphaRanker) -> None:
        self.ranker = ranker

    def get_dataframe(self) -> pd.DataFrame:
        """Get the full ranked leaderboard."""
        df = self.ranker.rank()
        if df.empty:
            return df

        # Add deployment readiness flag
        # Rule: Sharpe > 0.5, IC > 0.01, Hit Rate > 0.50
        def is_ready(row):
            return (
                row.get("sharpe_ratio", 0) > 0.5 and
                row.get("information_coefficient", 0) > 0.01 and
                row.get("hit_rate", 0) > 0.50
            )

        df["deployment_ready"] = df.apply(is_ready, axis=1)
        return df

    def get_top_k(self, k: int = 5) -> pd.DataFrame:
        """Get the top K alphas."""
        return self.get_dataframe().head(k)

    def get_pareto_frontier(self) -> list[str]:
        """Get Pareto-optimal alphas (Sharpe vs Max DD)."""
        return self.ranker.pareto_frontier(
            objectives=("sharpe_ratio", "max_drawdown_pct")
        )

    def summary_stats(self) -> dict[str, float]:
        """Aggregate stats across all evaluated alphas."""
        df = self.get_dataframe()
        if df.empty:
            return {}

        return {
            "count": len(df),
            "ready_count": int(df["deployment_ready"].sum()),
            "median_sharpe": float(df["sharpe_ratio"].median()),
            "max_sharpe": float(df["sharpe_ratio"].max()),
            "median_ic": float(df["information_coefficient"].median()),
            "max_ic": float(df["information_coefficient"].max()),
        }
