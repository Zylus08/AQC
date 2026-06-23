"""
aqc/research/tournament/alpha_reports.py
==========================================
Research report generator for Alpha Tournament.

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from aqc.research.tournament.alpha_leaderboard import AlphaLeaderboard

logger = logging.getLogger(__name__)


class TournamentReportGenerator:
    """Generates PDF/CSV research reports for the tournament."""

    def __init__(self, leaderboard: AlphaLeaderboard, output_dir: str = "reports/tournament"):
        self.leaderboard = leaderboard
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_csv(self, filename: str = "leaderboard.csv") -> None:
        """Export the leaderboard to CSV."""
        df = self.leaderboard.get_dataframe()
        if not df.empty:
            df.to_csv(self.output_dir / filename, index=False)
            logger.info("Tournament CSV saved to %s", self.output_dir / filename)

    def generate_markdown_report(self, filename: str = "tournament_report.md") -> None:
        """Generate a markdown summary report of the tournament."""
        df = self.leaderboard.get_dataframe()
        stats = self.leaderboard.summary_stats()
        pareto = self.leaderboard.get_pareto_frontier()

        lines = [
            "# AQC Alpha Tournament Report",
            "",
            "## Summary Statistics",
            f"- **Total Alphas Evaluated**: {stats.get('count', 0)}",
            f"- **Deployment Ready**: {stats.get('ready_count', 0)}",
            f"- **Median Sharpe**: {stats.get('median_sharpe', 0):.2f}",
            f"- **Max Sharpe**: {stats.get('max_sharpe', 0):.2f}",
            "",
            "## Pareto Frontier (Sharpe vs Drawdown)",
            *["- " + a for a in pareto],
            "",
            "## Top 5 Alphas",
            df.head(5).to_markdown(index=False),
        ]

        with open(self.output_dir / filename, "w") as f:
            f.write("\n".join(lines))
        
        logger.info("Tournament MD report saved to %s", self.output_dir / filename)
