"""
aqc/research/stress_testing/stress_reports.py
===============================================
Generates visualizations for Stress Test results.

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        return plt, sns
    except ImportError as exc:
        raise ImportError("matplotlib required. pip install matplotlib") from exc


class StressTestReports:
    """Visualizes stress testing outputs."""

    def __init__(self, output_dir: str = "reports/stress_tests") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_monte_carlo_distribution(self, mc_df: pd.DataFrame, alpha_name: str, save: bool = True):
        """Plot the distribution of Sharpe ratios from Monte Carlo."""
        plt, sns = _require_matplotlib()
        plt.style.use("dark_background")

        if mc_df.empty:
            return None

        fig, ax = plt.subplots(figsize=(10, 6), facecolor="#0d1117")
        ax.set_facecolor("#161b22")

        sns.histplot(mc_df["Sharpe"], bins=30, kde=True, color="#CE93D8", ax=ax, edgecolor="#30363d")

        # Add percentiles
        p5 = mc_df["Sharpe"].quantile(0.05)
        p50 = mc_df["Sharpe"].median()
        
        ax.axvline(p5, color="#F06292", linestyle="--", label=f"5th %ile ({p5:.2f})")
        ax.axvline(p50, color="#81C784", linestyle="--", label=f"Median ({p50:.2f})")

        ax.set_title(f"Monte Carlo Sharpe Distribution: {alpha_name}", color="white")
        ax.set_xlabel("Sharpe Ratio", color="white")
        ax.set_ylabel("Frequency", color="white")
        ax.tick_params(colors="white")
        ax.legend(facecolor="#21262d", labelcolor="white")

        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

        plt.tight_layout()

        if save:
            fig.savefig(self.output_dir / f"{alpha_name}_mc_sharpe.png", facecolor=fig.get_facecolor())

        return fig

    def plot_parameter_stability(self, param_results: dict[str, pd.DataFrame], alpha_name: str, save: bool = True):
        """Plot Sharpe ratio across parameter perturbations."""
        if not param_results:
            return None

        plt, _ = _require_matplotlib()
        plt.style.use("dark_background")

        n_params = len(param_results)
        fig, axes = plt.subplots(n_params, 1, figsize=(10, 4 * n_params), facecolor="#0d1117", squeeze=False)
        
        for i, (param, df) in enumerate(param_results.items()):
            ax = axes[i, 0]
            ax.set_facecolor("#161b22")
            
            ax.plot(df["Parameter_Value"], df["Sharpe"], marker='o', color="#4FC3F7", linewidth=2)
            ax.set_title(f"{param} Stability", color="white")
            ax.set_xlabel("Parameter Value", color="white")
            ax.set_ylabel("Sharpe Ratio", color="white")
            ax.tick_params(colors="white")
            
            for spine in ax.spines.values():
                spine.set_edgecolor("#30363d")

        plt.tight_layout()

        if save:
            fig.savefig(self.output_dir / f"{alpha_name}_param_stability.png", facecolor=fig.get_facecolor())

        return fig
