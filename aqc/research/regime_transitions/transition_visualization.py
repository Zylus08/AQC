"""
aqc/research/regime_transitions/transition_visualization.py
=============================================================
Visualizations for regime transition alpha research.

Author: Saksham Mishra — AlgoQuant Club
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TransitionVisualizer:
    """Generate heatmaps and charts for transition analysis.

    Parameters
    ----------
    analyzer : TransitionAlphaAnalyzer
        Initialized analyzer with computed returns.
    """

    def __init__(self, analyzer: 'TransitionAlphaAnalyzer') -> None:
        self.analyzer = analyzer

    def plot_all(self, output_dir: str = "reports/plots") -> None:
        p = Path(output_dir)
        p.mkdir(parents=True, exist_ok=True)
        
        alpha_df = self.analyzer.analyze_alpha()
        if alpha_df.empty:
            return

        self.plot_frequency_matrix(p / "transition_frequency_matrix.png")
        self.plot_return_heatmap(alpha_df, "volatility", p / "transition_return_heatmap_vol.png")
        self.plot_return_heatmap(alpha_df, "trend", p / "transition_return_heatmap_trend.png")

        logger.info("Transition plots saved to %s", output_dir)

    def plot_frequency_matrix(self, save_path: Optional[Path] = None) -> None:
        """Plot transition counts matrix for volatility regimes."""
        freq = self.analyzer.transition_frequency_matrix("volatility")
        if freq.empty:
            return

        fig, ax = plt.subplots(figsize=(8, 6), facecolor="#0d1117")
        ax.set_facecolor("#161b22")

        data = freq.values
        im = ax.imshow(data, cmap="Blues", alpha=0.8)

        ax.set_xticks(np.arange(len(freq.columns)))
        ax.set_yticks(np.arange(len(freq.index)))
        ax.set_xticklabels(freq.columns)
        ax.set_yticklabels(freq.index)

        for i in range(len(freq.index)):
            for j in range(len(freq.columns)):
                val = data[i, j]
                color = "white" if val > np.max(data)*0.5 else "black"
                ax.text(j, i, str(val), ha="center", va="center", color=color, fontweight="bold")

        ax.set_title("Volatility Transition Frequency Matrix", color="white", fontweight="bold")
        ax.set_xlabel("To Regime", color="white")
        ax.set_ylabel("From Regime", color="white")
        self._format_axis(ax)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

    def plot_return_heatmap(self, alpha_df: pd.DataFrame, regime_type: str, save_path: Optional[Path] = None) -> None:
        """Plot forward returns heatmap across horizons."""
        df = alpha_df[alpha_df["regime_type"] == regime_type]
        if df.empty:
            return

        horizons = self.analyzer.horizons
        cols = [f"avg_ret_{h}d" for h in horizons if f"avg_ret_{h}d" in df.columns]
        
        data = df[cols].values * 100  # Convert to %
        labels = df["transition_pair"].tolist()

        fig, ax = plt.subplots(figsize=(10, len(labels)*0.5 + 2), facecolor="#0d1117")
        ax.set_facecolor("#161b22")

        # Center colormap around 0
        vmax = np.nanmax(np.abs(data))
        if np.isnan(vmax) or vmax == 0: vmax = 1.0
        im = ax.imshow(data, cmap="RdYlGn", aspect="auto", alpha=0.8, vmin=-vmax, vmax=vmax)

        ax.set_xticks(np.arange(len(cols)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels([f"{h}d" for h in horizons])
        ax.set_yticklabels(labels)

        for i in range(len(labels)):
            for j in range(len(cols)):
                val = data[i, j]
                if not np.isnan(val):
                    # Add significance star if available
                    pval = df.iloc[i].get(f"p_val_{horizons[j]}d", 1.0)
                    sig = "*" if pval < 0.05 else ""
                    color = "black" if abs(val) > vmax*0.4 else "white"
                    ax.text(j, i, f"{val:+.2f}%{sig}", ha="center", va="center", color=color, fontsize=9)

        ax.set_title(f"Forward Returns by {regime_type.capitalize()} Transition (* = p<0.05)", color="white", fontweight="bold")
        self._format_axis(ax)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

    def _format_axis(self, ax: plt.Axes) -> None:
        """Apply dark mode formatting to an axis."""
        for s in ax.spines.values():
            s.set_edgecolor("#30363d")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
