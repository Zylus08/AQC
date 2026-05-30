"""
aqc/research/comparison/reporting.py
======================================
Comparative Backtest Reporting & Visualisation.

Generates publication-quality plots and CSV reports for multi-variant
backtest comparisons.

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ComparisonReportGenerator:
    """Generate reports and plots for backtest comparisons.

    Parameters
    ----------
    comparator:
        A ``BacktestComparator`` with results loaded.
    regime_data:
        Optional DataFrame of regime labels (from ``RegimeEngine.detect_full_series``).
    output_dir:
        Directory for output files.

    Examples
    --------
    >>> reporter = ComparisonReportGenerator(comparator, regime_data=regimes)
    >>> reporter.generate_all()
    """

    # Dark-mode color palette
    COLORS = ["#4FC3F7", "#FF7043", "#66BB6A", "#AB47BC", "#FFB74D", "#EF5350"]
    BG_COLOR = "#0d1117"
    PANEL_COLOR = "#161b22"
    SPINE_COLOR = "#30363d"

    def __init__(
        self,
        comparator,
        regime_data: Optional[pd.DataFrame] = None,
        output_dir: str = "reports",
    ) -> None:
        self.comparator = comparator
        self.regime_data = regime_data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_all(self) -> None:
        """Generate all reports and plots."""
        self.save_comparison_csv()
        self.plot_equity_comparison()
        self.plot_drawdown_comparison()

        if self.regime_data is not None:
            self.plot_regime_timeline()
            self.plot_transition_matrix()

        self.plot_risk_contribution()
        logger.info("All comparison reports saved to %s", self.output_dir)

    # ------------------------------------------------------------------
    # CSV Reports
    # ------------------------------------------------------------------

    def save_comparison_csv(self, filename: str = "comparative_backtest_report.csv") -> pd.DataFrame:
        """Save side-by-side metrics comparison to CSV."""
        df = self.comparator.compare()
        path = self.output_dir / filename
        df.to_csv(path)
        logger.info("Saved comparison CSV: %s", path)
        return df

    def save_regime_report(self, filename: str = "regime_detection_report.csv") -> None:
        """Save regime detection data to CSV."""
        if self.regime_data is not None:
            path = self.output_dir / filename
            self.regime_data.to_csv(path)
            logger.info("Saved regime report: %s", path)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def plot_equity_comparison(self, filename: str = "equity_comparison.png") -> None:
        """Plot overlaid equity curves for all variants."""
        import matplotlib.pyplot as plt

        curves = self.comparator.get_all_equity_curves()
        if curves.empty:
            return

        fig, ax = self._create_figure()

        for i, col in enumerate(curves.columns):
            color = self.COLORS[i % len(self.COLORS)]
            ax.plot(curves.index, curves[col], label=col, color=color,
                    linewidth=1.5 if i == 0 else 1.2, alpha=0.9)

        ax.set_title("Equity Curve Comparison", color="white",
                      fontsize=14, fontweight="bold")
        ax.set_ylabel("Portfolio Value ($)", color="white", fontsize=11)
        ax.legend(fontsize=10, facecolor="#21262d", labelcolor="white")
        self._style_axis(ax)

        self._save_plot(fig, filename)

    def plot_drawdown_comparison(self, filename: str = "drawdown_comparison.png") -> None:
        """Plot drawdown curves for all variants."""
        import matplotlib.pyplot as plt

        curves = self.comparator.get_all_equity_curves()
        if curves.empty:
            return

        fig, ax = self._create_figure()

        for i, col in enumerate(curves.columns):
            eq = curves[col].dropna()
            if len(eq) < 2:
                continue
            running_max = eq.cummax()
            dd = (eq - running_max) / running_max * 100.0
            color = self.COLORS[i % len(self.COLORS)]
            ax.fill_between(dd.index, dd, 0, alpha=0.25, color=color)
            ax.plot(dd.index, dd, label=col, color=color, linewidth=1.2)

        ax.set_title("Drawdown Comparison", color="white",
                      fontsize=14, fontweight="bold")
        ax.set_ylabel("Drawdown (%)", color="white", fontsize=11)
        ax.legend(fontsize=10, facecolor="#21262d", labelcolor="white")
        self._style_axis(ax)

        self._save_plot(fig, filename)

    def plot_regime_timeline(self, filename: str = "regime_timeline.png") -> None:
        """Plot regime classification over time."""
        import matplotlib.pyplot as plt

        if self.regime_data is None or self.regime_data.empty:
            return

        fig, axes = plt.subplots(
            3, 1, figsize=(14, 10), facecolor=self.BG_COLOR, sharex=True,
        )

        regime_configs = [
            ("vol_regime", "Volatility Regime",
             {"LOW": 0, "NORMAL": 1, "HIGH": 2, "EXTREME": 3},
             {"LOW": "#1B5E20", "NORMAL": "#1565C0", "HIGH": "#E65100", "EXTREME": "#B71C1C"}),
            ("trend_regime", "Trend Regime",
             {"STRONG_DOWNTREND": 0, "DOWNTREND": 1, "RANGE_BOUND": 2, "UPTREND": 3, "STRONG_UPTREND": 4},
             {"STRONG_DOWNTREND": "#B71C1C", "DOWNTREND": "#E65100", "RANGE_BOUND": "#1565C0",
              "UPTREND": "#2E7D32", "STRONG_UPTREND": "#1B5E20"}),
            ("corr_regime", "Correlation Regime",
             {"LOW_CORRELATION": 0, "NORMAL_CORRELATION": 1, "HIGH_CORRELATION": 2, "CRISIS_CORRELATION": 3},
             {"LOW_CORRELATION": "#1B5E20", "NORMAL_CORRELATION": "#1565C0",
              "HIGH_CORRELATION": "#E65100", "CRISIS_CORRELATION": "#B71C1C"}),
        ]

        for ax, (col, title, val_map, color_map) in zip(axes, regime_configs):
            ax.set_facecolor(self.PANEL_COLOR)
            if col not in self.regime_data.columns:
                continue

            values = self.regime_data[col].map(val_map).fillna(1)
            colors = [color_map.get(v, "#1565C0") for v in self.regime_data[col]]
            ax.bar(self.regime_data.index, values, color=colors, width=1.0, alpha=0.8)
            ax.set_ylabel(title, color="white", fontsize=10)
            ax.set_yticks(list(val_map.values()))
            ax.set_yticklabels(list(val_map.keys()), fontsize=7, color="white")

            for spine in ax.spines.values():
                spine.set_edgecolor(self.SPINE_COLOR)
            ax.tick_params(colors="white")

        fig.suptitle("Market Regime Timeline", color="white",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        self._save_plot(fig, filename)

    def plot_transition_matrix(self, filename: str = "transition_matrix.png") -> None:
        """Plot regime transition probability heatmap."""
        import matplotlib.pyplot as plt
        from aqc.regimes.regime_engine import RegimeEngine

        if self.regime_data is None or "vol_regime" not in self.regime_data.columns:
            return

        engine = RegimeEngine(enable_hmm=False)
        transmat = engine.compute_transition_matrix(self.regime_data["vol_regime"])

        fig, ax = plt.subplots(figsize=(8, 6), facecolor=self.BG_COLOR)
        ax.set_facecolor(self.PANEL_COLOR)

        im = ax.imshow(transmat.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

        ax.set_xticks(range(len(transmat.columns)))
        ax.set_yticks(range(len(transmat.index)))
        ax.set_xticklabels(transmat.columns, fontsize=9, color="white", rotation=45, ha="right")
        ax.set_yticklabels(transmat.index, fontsize=9, color="white")

        # Annotate cells
        for i in range(len(transmat.index)):
            for j in range(len(transmat.columns)):
                val = transmat.values[i, j]
                color = "white" if val < 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=11, color=color, fontweight="bold")

        cb = fig.colorbar(im, ax=ax, shrink=0.8)
        cb.ax.tick_params(colors="white")
        cb.set_label("Transition Probability", color="white", fontsize=10)

        ax.set_title("Volatility Regime Transition Matrix", color="white",
                      fontsize=13, fontweight="bold")
        ax.set_xlabel("To", color="white", fontsize=11)
        ax.set_ylabel("From", color="white", fontsize=11)

        for spine in ax.spines.values():
            spine.set_edgecolor(self.SPINE_COLOR)

        plt.tight_layout()
        self._save_plot(fig, filename)

    def plot_risk_contribution(self, filename: str = "risk_contribution.png") -> None:
        """Plot comparative risk metrics bar chart."""
        import matplotlib.pyplot as plt

        comparison = self.comparator.compare()
        if comparison.empty:
            return

        risk_metrics = ["sharpe_ratio", "sortino_ratio", "max_drawdown_pct",
                        "annualised_volatility", "calmar_ratio", "win_rate"]
        available = [m for m in risk_metrics if m in comparison.index]

        if not available:
            return

        fig, axes = plt.subplots(
            2, 3, figsize=(16, 9), facecolor=self.BG_COLOR,
        )
        axes = axes.flatten()

        for idx, metric in enumerate(available):
            if idx >= len(axes):
                break
            ax = axes[idx]
            ax.set_facecolor(self.PANEL_COLOR)

            values = comparison.loc[metric]
            colors = [self.COLORS[i % len(self.COLORS)] for i in range(len(values))]

            bars = ax.bar(range(len(values)), values.values, color=colors, alpha=0.85)
            ax.set_xticks(range(len(values)))
            ax.set_xticklabels(values.index, fontsize=8, color="white", rotation=30, ha="right")
            ax.set_title(metric.replace("_", " ").title(), color="white",
                          fontsize=11, fontweight="bold")
            ax.tick_params(colors="white")

            for spine in ax.spines.values():
                spine.set_edgecolor(self.SPINE_COLOR)

        # Hide unused axes
        for idx in range(len(available), len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle("Risk Metric Comparison", color="white",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        self._save_plot(fig, filename)

    # ------------------------------------------------------------------
    # Plot helpers
    # ------------------------------------------------------------------

    def _create_figure(self, figsize=(14, 7)):
        """Create a dark-themed figure."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize, facecolor=self.BG_COLOR)
        ax.set_facecolor(self.PANEL_COLOR)
        return fig, ax

    def _style_axis(self, ax) -> None:
        """Apply dark theme styling."""
        for spine in ax.spines.values():
            spine.set_edgecolor(self.SPINE_COLOR)
        ax.tick_params(colors="white")
        ax.grid(axis="y", alpha=0.1, color="white")

    def _save_plot(self, fig, filename: str) -> None:
        """Save and close plot."""
        import matplotlib.pyplot as plt

        path = self.output_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        logger.info("Plot saved: %s", path)
        plt.close(fig)
