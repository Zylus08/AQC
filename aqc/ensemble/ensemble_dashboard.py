"""
aqc/ensemble/ensemble_dashboard.py
====================================
Dashboard for Ensemble Alpha Engine.

Visualizes dynamic weights, rolling composite performance, and
attribution of individual alphas to the ensemble.

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        return plt, gridspec
    except ImportError as exc:
        raise ImportError("matplotlib required. pip install matplotlib") from exc


class EnsembleDashboard:
    """Generate visual dashboards for the Ensemble Alpha Engine."""

    def __init__(self, output_dir: str = "reports/ensemble") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_dynamic_weights(
        self,
        weights_history: pd.DataFrame,
        save: bool = True,
        filename: str = "dynamic_weights.png",
    ):
        """Plot the evolution of alpha weights over time as an area chart."""
        plt, _ = _require_matplotlib()
        plt.style.use("dark_background")

        if weights_history.empty:
            return None

        fig, ax = plt.subplots(figsize=(12, 6), facecolor="#0d1117")
        ax.set_facecolor("#161b22")

        # Create stacked area chart
        x = weights_history.index
        y = [weights_history[col].values for col in weights_history.columns]
        labels = weights_history.columns
        
        # AQC Palette
        palette = ["#4FC3F7", "#81C784", "#FFB74D", "#F06292", "#CE93D8", "#80CBC4"]
        colors = [palette[i % len(palette)] for i in range(len(labels))]

        ax.stackplot(x, *y, labels=labels, colors=colors, alpha=0.8)

        ax.set_title("Dynamic Alpha Weights Over Time", color="white")
        ax.set_ylabel("Weight Allocation", color="white")
        ax.set_xlabel("Time", color="white")
        ax.margins(0, 0) # Tight bounds for area chart

        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), facecolor="#21262d", labelcolor="white")
        ax.tick_params(colors="white")

        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

        plt.tight_layout()

        if save:
            fig.savefig(self.output_dir / filename, facecolor=fig.get_facecolor(), bbox_inches="tight")

        return fig

    def plot_performance_attribution(
        self,
        signal_returns: pd.DataFrame,
        ensemble_returns: pd.Series,
        save: bool = True,
        filename: str = "ensemble_attribution.png",
    ):
        """Plot cumulative returns of the ensemble vs its components."""
        plt, _ = _require_matplotlib()
        plt.style.use("dark_background")

        fig, ax = plt.subplots(figsize=(12, 6), facecolor="#0d1117")
        ax.set_facecolor("#161b22")

        # Plot components
        palette = ["#4FC3F7", "#81C784", "#FFB74D", "#F06292", "#CE93D8", "#80CBC4"]
        for i, col in enumerate(signal_returns.columns):
            cum_ret = (1 + signal_returns[col]).cumprod() - 1
            ax.plot(cum_ret.index, cum_ret, label=col, color=palette[i % len(palette)], alpha=0.4, linewidth=1)

        # Plot ensemble
        ens_cum = (1 + ensemble_returns).cumprod() - 1
        ax.plot(ens_cum.index, ens_cum, label="Ensemble", color="white", linewidth=2.5)

        ax.set_title("Ensemble vs Component Cumulative Returns", color="white")
        ax.set_ylabel("Cumulative Return", color="white")
        ax.axhline(0, color="#555", linewidth=1, linestyle="--")

        ax.legend(loc="upper left", facecolor="#21262d", labelcolor="white")
        ax.tick_params(colors="white")

        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

        plt.tight_layout()

        if save:
            fig.savefig(self.output_dir / filename, facecolor=fig.get_facecolor())

        return fig
