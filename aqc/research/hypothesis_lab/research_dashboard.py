"""
aqc/research/hypothesis_lab/research_dashboard.py
===================================================
Visualizes hypothesis tracking and research funnels.

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from aqc.research.hypothesis_lab.hypothesis_tracker import HypothesisTracker

logger = logging.getLogger(__name__)


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError as exc:
        raise ImportError("matplotlib required. pip install matplotlib") from exc


class HypothesisDashboard:
    """Generates visualizations for the Hypothesis Lab."""

    def __init__(self, tracker: HypothesisTracker, output_dir: str = "reports/hypothesis_lab"):
        self.tracker = tracker
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_funnel(self, save: bool = True, filename: str = "hypothesis_funnel.png"):
        """Plot the research funnel (IDEA -> DEPLOYED)."""
        plt = _require_matplotlib()
        plt.style.use("dark_background")

        df = self.tracker.get_summary_dataframe()
        if df.empty:
            logger.warning("No data to plot.")
            return None

        # Count by status
        counts = df["Status"].value_counts()
        
        # Define funnel order
        order = ["IDEA", "TESTING", "PROMISING", "FAILED", "DEPLOYED", "RETIRED"]
        # Only plot statuses that exist
        ordered_counts = [counts.get(s, 0) for s in order]

        fig, ax = plt.subplots(figsize=(10, 6), facecolor="#0d1117")
        ax.set_facecolor("#161b22")

        # AQC Palette
        colors = ["#4FC3F7", "#FFB74D", "#81C784", "#F06292", "#CE93D8", "#80CBC4"]

        bars = ax.bar(order, ordered_counts, color=colors, alpha=0.8)

        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f'{int(height)}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom', color="white", weight='bold'
            )

        ax.set_title("Alpha Hypothesis Funnel", color="white", weight='bold', pad=20)
        ax.set_ylabel("Number of Hypotheses", color="white")
        
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

        plt.tight_layout()

        if save:
            fig.savefig(self.output_dir / filename, facecolor=fig.get_facecolor(), bbox_inches="tight")

        return fig
