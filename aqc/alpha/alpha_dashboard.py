"""
aqc/alpha/alpha_dashboard.py
==============================
Alpha Leaderboard & Health Dashboard.

Generates matplotlib figures showing the current state of all registered
alphas: ranked leaderboard, health status indicators, and key metrics.

Uses the AQC dark-theme palette consistent with existing dashboards.

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from aqc.alpha.alpha_base import AlphaBase, AlphaMetrics
from aqc.alpha.alpha_monitor import AlphaMonitor, HealthReport, HealthStatus
from aqc.alpha.alpha_ranker import AlphaRanker

logger = logging.getLogger(__name__)

# AQC Dark Theme Constants
BG_COLOR = "#0d1117"
PLOT_FACE = "#161b22"
BORDER_COLOR = "#30363d"
CARD_COLOR = "#21262d"
PALETTE = ["#4FC3F7", "#81C784", "#FFB74D", "#F06292", "#CE93D8", "#80CBC4"]
STATUS_COLORS = {
    HealthStatus.GREEN: "#81C784",
    HealthStatus.YELLOW: "#FFB74D",
    HealthStatus.RED: "#F06292",
    HealthStatus.CRITICAL: "#E53935",
}


def _require_matplotlib():
    """Import matplotlib and return plt."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        return plt, gridspec
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for dashboards.  pip install matplotlib"
        ) from exc


class AlphaDashboard:
    """Generate visual dashboards for alpha research.

    Parameters
    ----------
    output_dir:
        Directory to save plots.  Created if missing.

    Examples
    --------
    >>> dash = AlphaDashboard("reports/alpha")
    >>> fig = dash.plot_leaderboard(leaderboard_df)
    """

    def __init__(self, output_dir: str = "reports/alpha") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Leaderboard
    # ------------------------------------------------------------------

    def plot_leaderboard(
        self,
        leaderboard: pd.DataFrame,
        save: bool = True,
        filename: str = "alpha_leaderboard.png",
        figsize: tuple[int, int] = (16, 8),
    ):
        """Plot the alpha leaderboard as a horizontal bar chart.

        Parameters
        ----------
        leaderboard:
            DataFrame from :meth:`AlphaRanker.rank`.
        save:
            Whether to save to disk.
        filename:
            Output filename.
        figsize:
            Figure size.

        Returns
        -------
        matplotlib.figure.Figure
        """
        plt, gridspec = _require_matplotlib()
        plt.style.use("dark_background")

        if leaderboard.empty:
            fig, ax = plt.subplots(figsize=figsize, facecolor=BG_COLOR)
            ax.set_facecolor(PLOT_FACE)
            ax.text(0.5, 0.5, "No alphas to display", transform=ax.transAxes,
                    ha="center", va="center", color="white", fontsize=14)
            return fig

        fig = plt.figure(figsize=figsize, facecolor=BG_COLOR)
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2], wspace=0.3)

        # Left: composite score bar chart
        ax_bar = fig.add_subplot(gs[0])
        ax_bar.set_facecolor(PLOT_FACE)

        names = leaderboard["alpha_name"].tolist()
        scores = leaderboard["composite_score"].tolist()
        colors = [PALETTE[i % len(PALETTE)] for i in range(len(names))]

        y_pos = np.arange(len(names))
        ax_bar.barh(y_pos, scores, color=colors, alpha=0.85, height=0.6)
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels(names, color="white", fontsize=9)
        ax_bar.set_xlabel("Composite Score", color="white", fontsize=10)
        ax_bar.set_title("Alpha Rankings", color="white", fontsize=12, fontweight="bold")
        ax_bar.tick_params(colors="white")
        ax_bar.invert_yaxis()
        for spine in ax_bar.spines.values():
            spine.set_edgecolor(BORDER_COLOR)

        # Right: metrics table
        ax_table = fig.add_subplot(gs[1])
        ax_table.set_facecolor(PLOT_FACE)
        ax_table.axis("off")

        cols_to_show = [
            "rank", "alpha_name", "composite_score",
            "sharpe_ratio", "information_coefficient", "hit_rate",
            "max_drawdown_pct", "profit_factor",
        ]
        cols_available = [c for c in cols_to_show if c in leaderboard.columns]
        display_df = leaderboard[cols_available].copy()

        for col in display_df.columns:
            if display_df[col].dtype in ("float64", "float32"):
                display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")

        table = ax_table.table(
            cellText=display_df.values,
            colLabels=display_df.columns,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.4)

        for key, cell in table.get_celld().items():
            cell.set_edgecolor(BORDER_COLOR)
            if key[0] == 0:
                cell.set_facecolor("#30363d")
                cell.set_text_props(color="white", fontweight="bold")
            else:
                cell.set_facecolor(PLOT_FACE)
                cell.set_text_props(color="white")

        fig.suptitle(
            "AQC Alpha Leaderboard",
            fontsize=14, fontweight="bold", color="white", y=0.98,
        )
        plt.tight_layout()

        if save:
            path = self.output_dir / filename
            fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
            logger.info("Leaderboard plot saved to %s", path)

        return fig

    # ------------------------------------------------------------------
    # Health dashboard
    # ------------------------------------------------------------------

    def plot_health_dashboard(
        self,
        reports: list[HealthReport],
        save: bool = True,
        filename: str = "alpha_health.png",
        figsize: tuple[int, int] = (14, 6),
    ):
        """Plot health status indicators for all monitored alphas.

        Parameters
        ----------
        reports:
            List of latest health reports (one per alpha).
        save:
            Whether to save to disk.
        filename:
            Output filename.
        figsize:
            Figure size.

        Returns
        -------
        matplotlib.figure.Figure
        """
        plt, gridspec = _require_matplotlib()
        plt.style.use("dark_background")

        fig, ax = plt.subplots(figsize=figsize, facecolor=BG_COLOR)
        ax.set_facecolor(PLOT_FACE)

        if not reports:
            ax.text(0.5, 0.5, "No health data", transform=ax.transAxes,
                    ha="center", va="center", color="white", fontsize=14)
            return fig

        names = [r.alpha_name for r in reports]
        ics = [r.rolling_ic for r in reports]
        hit_rates = [r.rolling_hit_rate for r in reports]
        statuses = [r.status for r in reports]
        status_colors = [STATUS_COLORS.get(s, "#888") for s in statuses]

        x = np.arange(len(names))
        width = 0.35

        bars_ic = ax.bar(x - width / 2, ics, width, label="Rolling IC",
                         color="#4FC3F7", alpha=0.8)
        bars_hr = ax.bar(x + width / 2, hit_rates, width, label="Hit Rate",
                         color="#81C784", alpha=0.8)

        # Status indicators as colored dots below the x-axis
        for i, (name, color) in enumerate(zip(names, status_colors)):
            ax.scatter(i, -0.05, color=color, s=200, zorder=10, marker="s")

        ax.set_xticks(x)
        ax.set_xticklabels(names, color="white", fontsize=9, rotation=30, ha="right")
        ax.set_ylabel("Value", color="white", fontsize=10)
        ax.set_title("Alpha Health Dashboard", color="white",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=9, facecolor=CARD_COLOR, labelcolor="white")
        ax.axhline(0, color="#555", linewidth=0.8)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER_COLOR)

        plt.tight_layout()

        if save:
            path = self.output_dir / filename
            fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
            logger.info("Health dashboard saved to %s", path)

        return fig

    # ------------------------------------------------------------------
    # IC Decay comparison
    # ------------------------------------------------------------------

    def plot_ic_decay_comparison(
        self,
        decay_data: dict[str, dict[int, float]],
        save: bool = True,
        filename: str = "alpha_ic_decay.png",
        figsize: tuple[int, int] = (12, 6),
    ):
        """Plot IC decay curves for multiple alphas.

        Parameters
        ----------
        decay_data:
            ``{alpha_name: {lag: IC, ...}, ...}``
        save:
            Whether to save to disk.
        filename:
            Output filename.
        figsize:
            Figure size.

        Returns
        -------
        matplotlib.figure.Figure
        """
        plt, _ = _require_matplotlib()
        plt.style.use("dark_background")

        fig, ax = plt.subplots(figsize=figsize, facecolor=BG_COLOR)
        ax.set_facecolor(PLOT_FACE)

        for i, (name, ic_by_lag) in enumerate(decay_data.items()):
            if not ic_by_lag:
                continue
            lags = sorted(ic_by_lag.keys())
            ics = [ic_by_lag[lag] for lag in lags]
            color = PALETTE[i % len(PALETTE)]
            ax.plot(lags, ics, marker="o", markersize=3, color=color,
                    linewidth=1.5, label=name, alpha=0.85)

        ax.axhline(0, color="#555", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Lag (bars)", color="white", fontsize=10)
        ax.set_ylabel("Information Coefficient", color="white", fontsize=10)
        ax.set_title("Alpha IC Decay Comparison", color="white",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=9, facecolor=CARD_COLOR, labelcolor="white")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER_COLOR)

        plt.tight_layout()

        if save:
            path = self.output_dir / filename
            fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
            logger.info("IC decay plot saved to %s", path)

        return fig
