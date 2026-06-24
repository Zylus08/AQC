"""
aqc/research/features/feature_dashboard.py
============================================
Visualizes feature importance and ranking.

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from aqc.research.features.feature_ranker import FeatureRanker

logger = logging.getLogger(__name__)


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        return plt, sns
    except ImportError as exc:
        raise ImportError("matplotlib required. pip install matplotlib") from exc


class FeatureDashboard:
    """Generates visualizations for the Feature Discovery Engine."""

    def __init__(self, output_dir: str = "reports/features"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_leaderboard(
        self, 
        ranked_features: pd.DataFrame, 
        save: bool = True, 
        filename: str = "feature_leaderboard.png"
    ):
        """Plot a horizontal bar chart of feature Composite Scores."""
        plt, sns = _require_matplotlib()
        plt.style.use("dark_background")

        if ranked_features.empty:
            logger.warning("No data to plot.")
            return None

        # Take top 20 if there are many
        df = ranked_features.head(20).sort_values("Composite_Score", ascending=True)

        fig, ax = plt.subplots(figsize=(10, 8), facecolor="#0d1117")
        ax.set_facecolor("#161b22")

        bars = ax.barh(df.index, df["Composite_Score"], color="#4FC3F7", alpha=0.8)

        ax.set_title("Feature Leaderboard (Top 20)", color="white", weight='bold')
        ax.set_xlabel("Composite Score (IC + Stability)", color="white")
        
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

        plt.tight_layout()

        if save:
            fig.savefig(self.output_dir / filename, facecolor=fig.get_facecolor(), bbox_inches="tight")

        return fig
