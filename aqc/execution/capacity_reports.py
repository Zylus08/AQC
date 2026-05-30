"""
aqc/execution/capacity_reports.py
===================================
Generate reports and visualizations for capacity analysis.

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


class CapacityReportGenerator:
    """Generate reports and plots for capacity analysis.

    Parameters
    ----------
    capacity_df : pd.DataFrame
        Output of CapacityAnalyzer.run_capacity_analysis().
    max_capital : float
        Output of CapacityAnalyzer.detect_capacity_breakpoint().
    """

    def __init__(self, capacity_df: pd.DataFrame, max_capital: float) -> None:
        self.df = capacity_df
        self.max_capital = max_capital

    def build_report(self) -> str:
        if self.df.empty:
            return "No capacity data available."

        sep = "=" * 80
        thin = "-" * 80
        lines = ["", sep, "  AQC CAPACITY ANALYSIS REPORT", sep, ""]

        lines.append(f"  Recommended Maximum Deployable Capital : ${self.max_capital:,.0f}")
        lines.append("")

        lines += ["  PERFORMANCE DECAY BY AUM", thin]
        header = f"  {'Capital ($)':15s} | {'Sharpe':>8s} | {'CAGR':>8s} | {'Max DD':>8s} | {'Cost (bps)':>10s}"
        lines.append(header)
        lines.append(thin)

        for cap, row in self.df.iterrows():
            cagr_str = f"{row.get('cagr', 0)*100:>7.2f}%"
            mdd_str = f"{row.get('max_drawdown', 0)*100:>7.2f}%"
            lines.append(
                f"  {cap:>15,.0f} | {row.get('sharpe', 0):>8.2f} | {cagr_str} | {mdd_str} | {row.get('cost_bps', 0):>10.2f}"
            )
        
        lines.append("")
        lines += [sep, ""]
        return "\n".join(lines)

    def print_report(self) -> None:
        print(self.build_report())

    def save_csv(self, output_dir: str = "reports") -> None:
        p = Path(output_dir)
        p.mkdir(parents=True, exist_ok=True)
        if not self.df.empty:
            self.df.to_csv(p / "capacity_analysis_report.csv", index=True)
            logger.info("Capacity report saved to %s", output_dir)

    def plot_all(self, output_dir: str = "reports/plots") -> None:
        p = Path(output_dir)
        p.mkdir(parents=True, exist_ok=True)

        if self.df.empty:
            return

        self.plot_sharpe_decay(p / "sharpe_vs_capital.png")
        self.plot_cagr_decay(p / "cagr_vs_capital.png")
        self.plot_cost_curve(p / "execution_cost_curve.png")

        logger.info("Capacity plots saved to %s", output_dir)

    def plot_sharpe_decay(self, save_path: Optional[Path] = None) -> None:
        fig, ax = plt.subplots(figsize=(10, 6), facecolor="#0d1117")
        ax.set_facecolor("#161b22")

        caps = self.df.index.values
        sharpes = self.df["sharpe"].values

        ax.plot(caps, sharpes, color="#4FC3F7", marker="o", lw=2)
        ax.axvline(self.max_capital, color="#EF5350", ls="--", label=f"Max Capacity: ${self.max_capital:,.0f}")

        ax.set_xscale("log")
        ax.set_title("Alpha Decay: Sharpe Ratio vs Capital", color="white", fontweight="bold")
        ax.set_xlabel("Deployed Capital ($)", color="white")
        ax.set_ylabel("Sharpe Ratio", color="white")
        ax.legend(facecolor="#21262d", labelcolor="white")
        self._format_axis(ax)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

    def plot_cagr_decay(self, save_path: Optional[Path] = None) -> None:
        fig, ax = plt.subplots(figsize=(10, 6), facecolor="#0d1117")
        ax.set_facecolor("#161b22")

        caps = self.df.index.values
        cagr = self.df["cagr"].values * 100

        ax.plot(caps, cagr, color="#66BB6A", marker="s", lw=2)
        ax.axvline(self.max_capital, color="#EF5350", ls="--", label=f"Max Capacity: ${self.max_capital:,.0f}")

        ax.set_xscale("log")
        ax.set_title("Capacity Frontier: CAGR vs Capital", color="white", fontweight="bold")
        ax.set_xlabel("Deployed Capital ($)", color="white")
        ax.set_ylabel("CAGR (%)", color="white")
        ax.legend(facecolor="#21262d", labelcolor="white")
        self._format_axis(ax)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

    def plot_cost_curve(self, save_path: Optional[Path] = None) -> None:
        fig, ax = plt.subplots(figsize=(10, 6), facecolor="#0d1117")
        ax.set_facecolor("#161b22")

        caps = self.df.index.values
        cost_bps = self.df["cost_bps"].values

        ax.plot(caps, cost_bps, color="#FF7043", marker="^", lw=2)

        ax.set_xscale("log")
        ax.set_title("Execution Cost vs Capital", color="white", fontweight="bold")
        ax.set_xlabel("Deployed Capital ($)", color="white")
        ax.set_ylabel("Execution Cost (bps of AUM)", color="white")
        self._format_axis(ax)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

    def _format_axis(self, ax: plt.Axes) -> None:
        for s in ax.spines.values():
            s.set_edgecolor("#30363d")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
