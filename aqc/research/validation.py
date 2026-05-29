"""
aqc/research/validation.py
===========================
Post-WFO statistical validation and visualisation.

The :class:`WalkForwardValidator` takes a completed
:class:`~aqc.research.walk_forward.WalkForwardResult` and:

* Computes stability metrics (IS/OOS correlation, parameter CV)
* Produces publication-quality plots (equity curves, Sharpe distribution,
  parameter heatmaps)
* Generates a comprehensive text report

All plotting functions return ``matplotlib.figure.Figure`` objects so they
can be embedded in notebooks, saved to disk, or displayed interactively.

Plotting requires ``matplotlib`` (optional dependency).  When matplotlib is
not installed, the plot methods raise :class:`ImportError` with a clear
message.

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from aqc.research.walk_forward import WalkForwardResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: safe matplotlib import
# ---------------------------------------------------------------------------


def _require_matplotlib():
    """Import matplotlib and return (plt, mpl).  Raise if not installed."""
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import matplotlib.ticker as mticker
        return plt, matplotlib, gridspec, mticker
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plotting. Install it with:\n"
            "  pip install matplotlib"
        ) from exc


# ---------------------------------------------------------------------------
# WalkForwardValidator
# ---------------------------------------------------------------------------


class WalkForwardValidator:
    """Statistical validation and visualisation of walk-forward results.

    Parameters
    ----------
    result:
        A completed :class:`~aqc.research.walk_forward.WalkForwardResult`.
    output_dir:
        Directory to save plots and reports.  Created if it does not exist.

    Examples
    --------
    >>> validator = WalkForwardValidator(result=wf_result, output_dir="reports")
    >>> validator.print_report()
    >>> validator.plot_equity_curves(save=True)
    >>> validator.plot_sharpe_distribution(save=True)
    >>> validator.plot_parameter_stability(save=True)
    """

    def __init__(
        self,
        result: WalkForwardResult,
        output_dir: str = "reports",
    ) -> None:
        self.result = result
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Text Report
    # ------------------------------------------------------------------

    def print_report(self) -> None:
        """Print a comprehensive walk-forward validation report to stdout."""
        report = self.build_report_text()
        print(report)
        logger.info(report)

    def save_report(self, filename: str = "walk_forward_report.txt") -> Path:
        """Save the validation report to a text file.

        Parameters
        ----------
        filename:
            Output filename relative to ``output_dir``.

        Returns
        -------
        Path
            Absolute path to the saved file.
        """
        path = self.output_dir / filename
        report = self.build_report_text()
        path.write_text(report, encoding="utf-8")
        logger.info("Walk-forward report saved to %s", path)
        return path

    def build_report_text(self) -> str:
        """Build the full validation report as a string.

        Returns
        -------
        str
        """
        r = self.result
        agg = r.aggregate_metrics()
        stability = r.parameter_stability()
        is_oos = self.is_oos_correlation()

        sep = "=" * 68
        thin = "-" * 68
        lines = [
            "",
            sep,
            "  AQC WALK-FORWARD OPTIMISATION REPORT",
            sep,
            "",
            f"  Mode             : {r.mode.name}",
            f"  Objective        : {r.objective_metric.value}",
            f"  Folds            : {agg['n_folds']}",
            f"  Total Time       : {agg['total_elapsed_seconds']:.1f}s",
            "",
            thin,
            "  PER-FOLD RESULTS",
            thin,
            "",
        ]

        # Header
        lines.append(
            f"  {'Fold':>4}  {'Train Sharpe':>12}  {'Test Sharpe':>11}  "
            f"{'Train Ret%':>10}  {'Test Ret%':>9}  {'Best Params'}"
        )
        lines.append(f"  {'-'*4}  {'-'*12}  {'-'*11}  {'-'*10}  {'-'*9}  {'-'*30}")

        for fold in r.folds:
            ts = fold.train_metrics.get("sharpe_ratio", float("nan"))
            oos = fold.test_metrics.get("sharpe_ratio", float("nan"))
            tr = fold.train_metrics.get("total_return_pct", float("nan"))
            ter = fold.test_metrics.get("total_return_pct", float("nan"))
            ts_str = f"{ts:>12.4f}" if math.isfinite(ts) else f"{'N/A':>12}"
            oos_str = f"{oos:>11.4f}" if math.isfinite(oos) else f"{'N/A':>11}"
            tr_str = f"{tr:>10.2f}" if math.isfinite(tr) else f"{'N/A':>10}"
            ter_str = f"{ter:>9.2f}" if math.isfinite(ter) else f"{'N/A':>9}"
            lines.append(
                f"  {fold.fold_index:>4}  {ts_str}  {oos_str}  "
                f"{tr_str}  {ter_str}  {fold.best_params}"
            )

        lines.extend([
            "",
            thin,
            "  AGGREGATE OUT-OF-SAMPLE STATISTICS",
            thin,
            "",
        ])

        metrics_to_show = [
            ("Sharpe Ratio",    "test_sharpe_ratio"),
            ("Sortino Ratio",   "test_sortino_ratio"),
            ("CAGR",            "test_cagr"),
            ("Max Drawdown %",  "test_max_drawdown_pct"),
            ("Win Rate",        "test_win_rate"),
            ("Profit Factor",   "test_profit_factor"),
            ("Total Return %",  "test_total_return_pct"),
        ]

        for label, key in metrics_to_show:
            mean = agg.get(f"{key}_mean", float("nan"))
            std = agg.get(f"{key}_std", float("nan"))
            mn = agg.get(f"{key}_min", float("nan"))
            mx = agg.get(f"{key}_max", float("nan"))
            if math.isfinite(mean):
                lines.append(
                    f"  {label:<20}: mean={mean:>8.4f}  std={std:>7.4f}  "
                    f"min={mn:>8.4f}  max={mx:>8.4f}"
                )
            else:
                lines.append(f"  {label:<20}: N/A")

        lines.extend([
            "",
            thin,
            "  IS/OOS CORRELATION",
            thin,
            "",
        ])

        if math.isfinite(is_oos):
            interpretation = (
                "Strong positive correlation — parameters generalise well."
                if is_oos > 0.6
                else "Moderate correlation — some overfitting present."
                if is_oos > 0.2
                else "Weak/negative correlation — potential overfitting."
            )
            lines.append(f"  IS/OOS Sharpe Correlation : {is_oos:>8.4f}")
            lines.append(f"  Interpretation            : {interpretation}")
        else:
            lines.append("  IS/OOS Sharpe Correlation : N/A (insufficient data)")

        lines.extend([
            "",
            thin,
            "  PARAMETER STABILITY",
            thin,
            "",
        ])

        for param_name, stats in stability.items():
            if "cv" in stats:
                cv_str = f"{stats['cv']:.4f}" if math.isfinite(stats["cv"]) else "N/A"
                lines.append(
                    f"  {param_name:<20}: mean={stats['mean']:>8.4f}  "
                    f"std={stats['std']:>7.4f}  cv={cv_str:>8}"
                    f"  values={stats['values_per_fold']}"
                )
            else:
                lines.append(
                    f"  {param_name:<20}: mode={stats.get('mode')}  "
                    f"values={stats['values_per_fold']}"
                )

        lines.extend(["", sep, ""])
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Statistical analysis
    # ------------------------------------------------------------------

    def is_oos_correlation(self) -> float:
        """Compute the Pearson correlation between IS and OOS Sharpe Ratios.

        A high positive correlation indicates that in-sample performance
        reliably predicts out-of-sample performance (low overfitting risk).

        Returns
        -------
        float
            Pearson r in ``[-1, 1]``, or ``nan`` if insufficient data.
        """
        is_sharpes = [
            f.train_metrics.get("sharpe_ratio", float("nan"))
            for f in self.result.folds
        ]
        oos_sharpes = [
            f.test_metrics.get("sharpe_ratio", float("nan"))
            for f in self.result.folds
        ]

        # Filter to pairs where both are finite
        pairs = [
            (is_s, oos_s)
            for is_s, oos_s in zip(is_sharpes, oos_sharpes)
            if math.isfinite(is_s) and math.isfinite(oos_s)
        ]

        if len(pairs) < 2:
            return float("nan")

        is_arr = np.array([p[0] for p in pairs])
        oos_arr = np.array([p[1] for p in pairs])

        if is_arr.std() < 1e-10 or oos_arr.std() < 1e-10:
            return float("nan")

        corr = float(np.corrcoef(is_arr, oos_arr)[0, 1])
        return round(corr, 4)

    def overfitting_score(self) -> float:
        """Compute an overfitting score (0 = no overfitting, 1 = heavy).

        Based on the ratio of IS return to OOS return.

        Returns
        -------
        float
            Overfitting score in ``[0, 1]``.
        """
        is_returns = [
            f.train_metrics.get("total_return_pct", float("nan"))
            for f in self.result.folds
        ]
        oos_returns = [
            f.test_metrics.get("total_return_pct", float("nan"))
            for f in self.result.folds
        ]

        pairs = [
            (i, o)
            for i, o in zip(is_returns, oos_returns)
            if math.isfinite(i) and math.isfinite(o) and i != 0
        ]

        if not pairs:
            return float("nan")

        ratios = [max(0.0, 1.0 - (o / i)) for i, o in pairs if i > 0]
        return round(float(np.mean(ratios)), 4) if ratios else float("nan")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def plot_equity_curves(
        self,
        save: bool = True,
        filename: str = "wfo_equity_curves.png",
        figsize: tuple[int, int] = (14, 8),
    ):
        """Plot test-window equity curves for each fold plus combined curve.

        Parameters
        ----------
        save:
            If ``True``, save to ``output_dir/filename``.
        filename:
            Output filename.
        figsize:
            Figure size in inches.

        Returns
        -------
        matplotlib.figure.Figure
        """
        plt, mpl, gridspec, mticker = _require_matplotlib()

        # Style
        plt.style.use("dark_background")
        PALETTE = [
            "#4FC3F7", "#81C784", "#FFB74D", "#F06292",
            "#CE93D8", "#80CBC4", "#FFCC80", "#EF9A9A",
        ]

        fig = plt.figure(figsize=figsize, facecolor="#0d1117")
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
        ax_main = fig.add_subplot(gs[0])
        ax_bar = fig.add_subplot(gs[1])

        fig.suptitle(
            "Walk-Forward Out-of-Sample Equity Curves",
            fontsize=14,
            fontweight="bold",
            color="white",
            y=0.98,
        )

        # Per-fold equity curves
        for fold in self.result.folds:
            curve = fold.test_equity_curve
            if curve.empty or "equity" not in curve.columns:
                continue
            color = PALETTE[fold.fold_index % len(PALETTE)]
            scaled = curve["equity"] / curve["equity"].iloc[0] * 100
            ax_main.plot(
                scaled.index,
                scaled.values,
                color=color,
                alpha=0.5,
                linewidth=1.2,
                label=f"Fold {fold.fold_index} (OOS Sharpe: {fold.test_sharpe:.2f})",
            )

        # Combined equity curve
        combined = self.result.combined_test_equity_curve()
        if not combined.empty:
            combined_scaled = combined["equity"] / combined["equity"].iloc[0] * 100
            ax_main.plot(
                combined_scaled.index,
                combined_scaled.values,
                color="white",
                linewidth=2.5,
                label="Combined OOS",
                zorder=10,
            )

        # Reference line at 100
        ax_main.axhline(100, color="#555", linewidth=1, linestyle="--", alpha=0.7)

        ax_main.set_facecolor("#161b22")
        ax_main.tick_params(colors="white", labelsize=9)
        ax_main.set_ylabel("Equity (rebased to 100)", color="white", fontsize=10)
        ax_main.spines["bottom"].set_visible(False)
        ax_main.legend(loc="upper left", fontsize=8, facecolor="#21262d", labelcolor="white")
        for spine in ax_main.spines.values():
            spine.set_edgecolor("#30363d")

        # Bar chart: OOS Sharpe per fold
        fold_ids = [f.fold_index for f in self.result.folds]
        sharpes = [
            f.test_sharpe if math.isfinite(f.test_sharpe) else 0.0
            for f in self.result.folds
        ]
        bar_colors = [
            "#81C784" if s > 0 else "#F06292"
            for s in sharpes
        ]
        ax_bar.bar(fold_ids, sharpes, color=bar_colors, alpha=0.85, width=0.6)
        ax_bar.axhline(0, color="#888", linewidth=1)
        ax_bar.set_facecolor("#161b22")
        ax_bar.tick_params(colors="white", labelsize=9)
        ax_bar.set_xlabel("Fold", color="white", fontsize=10)
        ax_bar.set_ylabel("OOS Sharpe", color="white", fontsize=9)
        ax_bar.set_xticks(fold_ids)
        for spine in ax_bar.spines.values():
            spine.set_edgecolor("#30363d")

        plt.tight_layout()

        if save:
            path = self.output_dir / filename
            fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
            logger.info("Equity curve plot saved to %s", path)

        return fig

    def plot_sharpe_distribution(
        self,
        save: bool = True,
        filename: str = "wfo_sharpe_distribution.png",
        figsize: tuple[int, int] = (12, 5),
    ):
        """Plot IS vs OOS Sharpe Ratio distributions.

        Parameters
        ----------
        save:
            If ``True``, save to ``output_dir/filename``.
        filename:
            Output filename.
        figsize:
            Figure size.

        Returns
        -------
        matplotlib.figure.Figure
        """
        plt, mpl, gridspec, mticker = _require_matplotlib()
        plt.style.use("dark_background")

        is_sharpes = [
            f.train_sharpe for f in self.result.folds if math.isfinite(f.train_sharpe)
        ]
        oos_sharpes = [
            f.test_sharpe for f in self.result.folds if math.isfinite(f.test_sharpe)
        ]

        fig, axes = plt.subplots(1, 2, figsize=figsize, facecolor="#0d1117")

        for ax, values, label, color in [
            (axes[0], is_sharpes, "In-Sample (Train)", "#4FC3F7"),
            (axes[1], oos_sharpes, "Out-of-Sample (Test)", "#81C784"),
        ]:
            ax.set_facecolor("#161b22")
            if values:
                ax.hist(values, bins=max(5, len(values) // 2), color=color, alpha=0.8, edgecolor="#21262d")
                mean_v = np.mean(values)
                ax.axvline(mean_v, color="white", linewidth=1.5, linestyle="--",
                           label=f"Mean: {mean_v:.3f}")
                ax.axvline(0, color="#888", linewidth=1, linestyle=":")
                ax.legend(fontsize=9, facecolor="#21262d", labelcolor="white")
            ax.set_title(f"Sharpe Distribution\n{label}", color="white", fontsize=11)
            ax.set_xlabel("Sharpe Ratio", color="white", fontsize=10)
            ax.set_ylabel("Count", color="white", fontsize=10)
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_edgecolor("#30363d")

        corr = self.is_oos_correlation()
        corr_str = f"IS/OOS Correlation: {corr:.3f}" if math.isfinite(corr) else "IS/OOS Correlation: N/A"
        fig.suptitle(
            f"Sharpe Ratio Distribution — {corr_str}",
            fontsize=13,
            fontweight="bold",
            color="white",
        )
        plt.tight_layout()

        if save:
            path = self.output_dir / filename
            fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
            logger.info("Sharpe distribution plot saved to %s", path)

        return fig

    def plot_parameter_stability(
        self,
        save: bool = True,
        filename: str = "wfo_parameter_stability.png",
        figsize: tuple[int, int] = (13, 6),
    ):
        """Plot parameter values chosen across folds as a heatmap / line plot.

        Parameters
        ----------
        save:
            If ``True``, save to ``output_dir/filename``.
        filename:
            Output filename.
        figsize:
            Figure size.

        Returns
        -------
        matplotlib.figure.Figure
        """
        plt, mpl, gridspec, mticker = _require_matplotlib()
        plt.style.use("dark_background")

        stability = self.result.parameter_stability()
        n_folds = len(self.result.folds)
        fold_ids = list(range(n_folds))

        numeric_params = {
            name: stats
            for name, stats in stability.items()
            if "mean" in stats
        }
        categorical_params = {
            name: stats
            for name, stats in stability.items()
            if "mode" in stats
        }

        n_numeric = len(numeric_params)
        if n_numeric == 0:
            logger.warning("No numeric parameters found for stability plot.")
            fig, ax = plt.subplots(facecolor="#0d1117")
            ax.set_facecolor("#161b22")
            ax.text(0.5, 0.5, "No numeric parameters", transform=ax.transAxes,
                    ha="center", va="center", color="white")
            return fig

        PALETTE = [
            "#4FC3F7", "#81C784", "#FFB74D", "#F06292",
            "#CE93D8", "#80CBC4", "#FFCC80",
        ]

        # Build heatmap matrix: rows = params, cols = folds
        param_names = list(numeric_params.keys())
        matrix = np.zeros((len(param_names), n_folds))

        for i, name in enumerate(param_names):
            raw_values = numeric_params[name]["values_per_fold"]
            for j, v in enumerate(raw_values):
                if isinstance(v, (int, float)) and math.isfinite(v):
                    matrix[i, j] = v

        # Normalise each row to [0, 1] for heatmap
        matrix_norm = np.zeros_like(matrix)
        for i in range(len(param_names)):
            row = matrix[i]
            rng = row.max() - row.min()
            if rng > 1e-10:
                matrix_norm[i] = (row - row.min()) / rng
            else:
                matrix_norm[i] = 0.5

        fig, axes = plt.subplots(1, 2, figsize=figsize, facecolor="#0d1117",
                                 gridspec_kw={"width_ratios": [1, 1.5]})

        # Left: heatmap
        ax_heat = axes[0]
        ax_heat.set_facecolor("#161b22")
        im = ax_heat.imshow(matrix_norm, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax_heat.set_yticks(range(len(param_names)))
        ax_heat.set_yticklabels(param_names, color="white", fontsize=9)
        ax_heat.set_xticks(fold_ids)
        ax_heat.set_xticklabels([f"F{i}" for i in fold_ids], color="white", fontsize=9)
        ax_heat.set_title("Parameter Values per Fold\n(normalised)", color="white", fontsize=11)
        ax_heat.tick_params(colors="white")
        for row_idx in range(len(param_names)):
            for col_idx in range(n_folds):
                val = matrix[row_idx, col_idx]
                ax_heat.text(
                    col_idx, row_idx,
                    f"{int(val)}" if isinstance(self.result.folds[0].best_params.get(param_names[row_idx]), int) else f"{val:.2f}",
                    ha="center", va="center", fontsize=7, color="white"
                )
        cbar = fig.colorbar(im, ax=ax_heat, shrink=0.8)
        cbar.ax.yaxis.set_tick_params(color="white")
        cbar.ax.tick_params(labelcolor="white", labelsize=8)

        # Right: line plot of absolute values
        ax_line = axes[1]
        ax_line.set_facecolor("#161b22")
        for i, name in enumerate(param_names):
            raw_values = numeric_params[name]["values_per_fold"]
            color = PALETTE[i % len(PALETTE)]
            ax_line.plot(
                fold_ids[:len(raw_values)],
                [v if isinstance(v, (int, float)) else float("nan") for v in raw_values],
                marker="o",
                color=color,
                linewidth=1.5,
                markersize=5,
                label=name,
            )
        ax_line.set_title("Parameter Trajectory", color="white", fontsize=11)
        ax_line.set_xlabel("Fold", color="white", fontsize=10)
        ax_line.set_ylabel("Parameter Value", color="white", fontsize=10)
        ax_line.tick_params(colors="white")
        ax_line.legend(fontsize=9, facecolor="#21262d", labelcolor="white")
        ax_line.set_xticks(fold_ids)
        ax_line.set_xticklabels([f"F{i}" for i in fold_ids], color="white")
        for spine in ax_line.spines.values():
            spine.set_edgecolor("#30363d")
        for spine in ax_heat.spines.values():
            spine.set_edgecolor("#30363d")

        fig.suptitle(
            "Parameter Stability Analysis",
            fontsize=13,
            fontweight="bold",
            color="white",
        )
        plt.tight_layout()

        if save:
            path = self.output_dir / filename
            fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
            logger.info("Parameter stability plot saved to %s", path)

        return fig

    def plot_all(self, save: bool = True) -> dict:
        """Generate all validation plots.

        Parameters
        ----------
        save:
            If ``True``, save each plot to disk.

        Returns
        -------
        dict
            ``{"equity_curves": fig, "sharpe_distribution": fig, "parameter_stability": fig}``
        """
        return {
            "equity_curves": self.plot_equity_curves(save=save),
            "sharpe_distribution": self.plot_sharpe_distribution(save=save),
            "parameter_stability": self.plot_parameter_stability(save=save),
        }

    def __repr__(self) -> str:
        return (
            f"WalkForwardValidator("
            f"folds={len(self.result.folds)}, "
            f"output_dir={self.output_dir})"
        )
