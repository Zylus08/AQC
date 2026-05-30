"""
aqc/diagnostics/trade_visualization.py
========================================
Visualization tools for trade-level attribution and PnL decomposition.

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


class TradeVisualizer:
    """Generate trade attribution plots and heatmaps.

    Parameters
    ----------
    attribution : dict[str, pd.DataFrame]
        From ``TradeAttributionEngine.full_summary()``.
    trades_df : pd.DataFrame
        From ``TradeAnalyzer.to_dataframe()``.
    """

    def __init__(
        self,
        attribution: dict[str, pd.DataFrame],
        trades_df: pd.DataFrame,
    ) -> None:
        self.attr = attribution
        self.df = trades_df

    def plot_all(self, output_dir: str = "reports/plots") -> None:
        """Generate all standard trade visualizations."""
        p = Path(output_dir)
        p.mkdir(parents=True, exist_ok=True)

        self.plot_pnl_by_regime(p / "pnl_by_regime.png")
        self.plot_pnl_by_duration(p / "pnl_by_duration.png")
        self.plot_pnl_by_signal(p / "pnl_by_signal.png")
        self.plot_regime_duration_heatmap(p / "regime_duration_heatmap.png")
        self.plot_trade_distribution(p / "trade_distribution.png")
        self.plot_mfe_mae_scatter(p / "mfe_mae_scatter.png")

        logger.info("Trade attribution plots saved to %s", output_dir)

    # ------------------------------------------------------------------
    # Plotting Methods
    # ------------------------------------------------------------------

    def plot_pnl_by_regime(self, save_path: Optional[Path] = None) -> None:
        """Bar chart of total PnL by volatility regime."""
        df = self.attr.get("by_vol_regime")
        if df is None or df.empty:
            return

        fig, ax = plt.subplots(figsize=(10, 6), facecolor="#0d1117")
        ax.set_facecolor("#161b22")

        labels = df["label"].tolist()
        vals = df["total_pnl"].values
        colors = ["#1B5E20" if v >= 0 else "#B71C1C" for v in vals]

        bars = ax.bar(labels, vals, color=colors, alpha=0.85, edgecolor="#21262d")
        ax.axhline(0, color="white", lw=0.8)

        for bar in bars:
            height = bar.get_height()
            offset = 10 if height >= 0 else -20
            ax.annotate(
                f"${height:,.0f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, offset),
                textcoords="offset points",
                ha="center", va="bottom", color="white", fontsize=9
            )

        ax.set_title("Total PnL by Volatility Regime", color="white", fontweight="bold")
        ax.set_ylabel("Realised PnL ($)", color="white")
        self._format_axis(ax)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

    def plot_pnl_by_duration(self, save_path: Optional[Path] = None) -> None:
        """Bar chart of PnL by holding duration bucket."""
        df = self.attr.get("by_duration")
        if df is None or df.empty:
            return

        # Sort logically if buckets match standard names
        order = {"0-1d": 0, "1-5d": 1, "5-20d": 2, "20+d": 3}
        df = df.copy()
        df["_sort"] = df["label"].map(lambda x: order.get(str(x), 99))
        df = df.sort_values("_sort")

        fig, ax = plt.subplots(figsize=(10, 6), facecolor="#0d1117")
        ax.set_facecolor("#161b22")

        labels = df["label"].tolist()
        vals = df["total_pnl"].values
        colors = ["#4FC3F7"] * len(vals)

        bars = ax.bar(labels, vals, color=colors, alpha=0.85, edgecolor="#21262d")
        ax.axhline(0, color="white", lw=0.8)

        for bar in bars:
            height = bar.get_height()
            offset = 5 if height >= 0 else -15
            ax.annotate(
                f"${height:,.0f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, offset),
                textcoords="offset points",
                ha="center", va="bottom", color="white", fontsize=9
            )

        ax.set_title("Total PnL by Holding Duration", color="white", fontweight="bold")
        ax.set_ylabel("Realised PnL ($)", color="white")
        self._format_axis(ax)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

    def plot_pnl_by_signal(self, save_path: Optional[Path] = None) -> None:
        """Bar chart of PnL by signal source."""
        df = self.attr.get("by_signal")
        if df is None or df.empty:
            return

        df = df.sort_values("total_pnl", ascending=True)

        fig, ax = plt.subplots(figsize=(10, 6), facecolor="#0d1117")
        ax.set_facecolor("#161b22")

        labels = df["label"].tolist()
        vals = df["total_pnl"].values
        colors = ["#66BB6A" if v >= 0 else "#EF5350" for v in vals]

        bars = ax.barh(labels, vals, color=colors, alpha=0.85, edgecolor="#21262d")
        ax.axvline(0, color="white", lw=0.8)

        for bar in bars:
            width = bar.get_width()
            offset = 5 if width >= 0 else -30
            ha = "left" if width >= 0 else "right"
            ax.annotate(
                f"${width:,.0f}",
                xy=(width, bar.get_y() + bar.get_height() / 2),
                xytext=(offset, 0),
                textcoords="offset points",
                ha=ha, va="center", color="white", fontsize=9
            )

        ax.set_title("Total PnL by Signal Source", color="white", fontweight="bold")
        ax.set_xlabel("Realised PnL ($)", color="white")
        self._format_axis(ax)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

    def plot_regime_duration_heatmap(self, save_path: Optional[Path] = None) -> None:
        """Heatmap of PnL: Vol Regime vs Duration Bucket."""
        df = self.attr.get("heatmap_regime_duration")
        if df is None or df.empty:
            return

        fig, ax = plt.subplots(figsize=(10, 6), facecolor="#0d1117")
        ax.set_facecolor("#161b22")

        # Basic heatmap
        data = df.values
        im = ax.imshow(data, cmap="RdYlGn", aspect="auto", alpha=0.8)
        
        # Labels
        ax.set_xticks(np.arange(len(df.columns)))
        ax.set_yticks(np.arange(len(df.index)))
        ax.set_xticklabels(df.columns)
        ax.set_yticklabels(df.index)

        # Annotations
        for i in range(len(df.index)):
            for j in range(len(df.columns)):
                val = data[i, j]
                text_color = "black" if abs(val) > np.max(np.abs(data))*0.4 else "white"
                ax.text(j, i, f"${val:,.0f}", ha="center", va="center", color=text_color, fontsize=9)

        ax.set_title("PnL Heatmap: Volatility Regime vs Duration", color="white", fontweight="bold")
        self._format_axis(ax)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

    def plot_trade_distribution(self, save_path: Optional[Path] = None) -> None:
        """Histogram of trade PnL."""
        if self.df.empty:
            return

        pnl = self.df["realised_pnl"].dropna()
        if pnl.empty:
            return

        fig, ax = plt.subplots(figsize=(10, 6), facecolor="#0d1117")
        ax.set_facecolor("#161b22")

        n, bins, patches = ax.hist(pnl, bins=50, edgecolor="#21262d", alpha=0.8)
        for i, patch in enumerate(patches):
            if bins[i] >= 0:
                patch.set_facecolor("#66BB6A")
            else:
                patch.set_facecolor("#EF5350")

        ax.axvline(0, color="white", lw=1.0, ls="--")
        ax.axvline(pnl.mean(), color="#4FC3F7", lw=1.5, ls="-", label=f"Mean: ${pnl.mean():.2f}")
        
        ax.set_title("Trade PnL Distribution", color="white", fontweight="bold")
        ax.set_xlabel("Realised PnL ($)", color="white")
        ax.legend(facecolor="#21262d", labelcolor="white")
        self._format_axis(ax)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

    def plot_mfe_mae_scatter(self, save_path: Optional[Path] = None) -> None:
        """Scatter plot of MFE vs MAE for trades."""
        if self.df.empty or "mfe_pct" not in self.df.columns:
            return

        fig, ax = plt.subplots(figsize=(10, 8), facecolor="#0d1117")
        ax.set_facecolor("#161b22")

        mfe = self.df["mfe_pct"] * 100
        mae = self.df["mae_pct"] * 100
        pnl = self.df["realised_pnl"]

        colors = ["#66BB6A" if p > 0 else "#EF5350" for p in pnl]
        sizes = [max(10, min(abs(p)/100, 200)) for p in pnl]  # Scale by absolute PnL

        ax.scatter(mae, mfe, c=colors, s=sizes, alpha=0.6, edgecolors="#21262d")
        
        # Diagonal line MFE = MAE
        max_val = max(mfe.max(), mae.max(), 1.0)
        ax.plot([0, max_val], [0, max_val], color="white", ls="--", lw=0.8, alpha=0.5)

        ax.set_title("Maximum Favourable vs Adverse Excursion (%)", color="white", fontweight="bold")
        ax.set_xlabel("MAE (%) - Max Drawdown during trade", color="white")
        ax.set_ylabel("MFE (%) - Max Runup during trade", color="white")
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
