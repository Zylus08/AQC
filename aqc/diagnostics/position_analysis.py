"""
aqc/diagnostics/position_analysis.py
======================================
Position size forensics: were positions excessively large? Was concentration driving performance?

Author: Saksham Mishra — AlgoQuant Club
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Optional
import numpy as np, pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class PositionStats:
    avg_size: float = 0.0
    max_size: float = 0.0
    avg_weight: float = 0.0
    max_weight: float = 0.0
    avg_num_positions: float = 0.0
    max_num_positions: int = 0
    hhi_concentration: float = 0.0
    avg_turnover: float = 0.0

class PositionAnalyzer:
    """Analyse position sizes, weights, concentration, and turnover.

    Parameters
    ----------
    equity_series : pd.Series
    position_values : pd.DataFrame  cols=symbols, vals=signed market value
    trade_log : list[dict]         from Portfolio.trade_log
    """
    def __init__(self, equity_series: pd.Series,
                 position_values: pd.DataFrame,
                 trade_log: Optional[list[dict]] = None) -> None:
        self.equity = equity_series
        self.pos_values = position_values
        self.trade_log = trade_log or []
        self._series: Optional[pd.DataFrame] = None

    def compute(self) -> pd.DataFrame:
        eq = self.equity.reindex(self.pos_values.index).ffill().bfill().replace(0, np.nan).ffill()
        weights = self.pos_values.abs().div(eq, axis=0)
        n_pos = (self.pos_values.abs() > 1e-6).sum(axis=1)
        max_weight = weights.max(axis=1)
        hhi = (weights ** 2).sum(axis=1)
        turnover = weights.diff().abs().sum(axis=1)
        df = pd.DataFrame({
            "num_positions": n_pos,
            "max_weight": max_weight,
            "hhi": hhi,
            "daily_turnover": turnover,
            "gross_value": self.pos_values.abs().sum(axis=1),
        }, index=self.pos_values.index)
        self._series = df
        return df

    def stats(self) -> PositionStats:
        df = self._series if self._series is not None else self.compute()
        eq = self.equity.reindex(self.pos_values.index).ffill().bfill().replace(0, np.nan).ffill()
        sizes = self.pos_values.abs().values.flatten()
        sizes = sizes[sizes > 1e-6]
        weights_flat = (self.pos_values.abs().div(eq, axis=0)).values.flatten()
        weights_flat = weights_flat[~np.isnan(weights_flat) & (weights_flat > 1e-6)]
        return PositionStats(
            avg_size=round(float(np.mean(sizes)), 2) if len(sizes) > 0 else 0.0,
            max_size=round(float(np.max(sizes)), 2) if len(sizes) > 0 else 0.0,
            avg_weight=round(float(np.mean(weights_flat)), 4) if len(weights_flat) > 0 else 0.0,
            max_weight=round(float(np.max(weights_flat)), 4) if len(weights_flat) > 0 else 0.0,
            avg_num_positions=round(float(df["num_positions"].mean()), 2),
            max_num_positions=int(df["num_positions"].max()),
            hhi_concentration=round(float(df["hhi"].mean()), 4),
            avg_turnover=round(float(df["daily_turnover"].mean()), 4),
        )

    def largest_positions(self, top_n: int = 10) -> pd.DataFrame:
        max_vals = self.pos_values.abs().max().sort_values(ascending=False).head(top_n)
        return pd.DataFrame({"symbol": max_vals.index, "max_market_value": max_vals.values})

    def save_report(self, path: str = "reports/position_analysis.csv") -> None:
        from pathlib import Path
        df = self._series if self._series is not None else self.compute()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path)

    def plot(self, save: bool = True, output_dir: str = "reports") -> None:
        import matplotlib.pyplot as plt
        from pathlib import Path
        df = self._series if self._series is not None else self.compute()
        fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor="#0d1117")
        for ax in axes.flatten(): ax.set_facecolor("#161b22")

        ax = axes[0, 0]
        sizes = self.pos_values.abs().values.flatten()
        sizes = sizes[sizes > 1e-6]
        if len(sizes) > 0:
            ax.hist(sizes, bins=40, color="#4FC3F7", alpha=0.8, edgecolor="#21262d")
        ax.set_title("Position Size Histogram", color="white", fontweight="bold")

        ax = axes[0, 1]
        ax.plot(df.index, df["max_weight"], color="#FF7043", lw=1.0)
        ax.set_title("Max Position Weight Over Time", color="white", fontweight="bold")

        ax = axes[1, 0]
        ax.plot(df.index, df["hhi"], color="#AB47BC", lw=1.0)
        ax.set_title("Portfolio Concentration (HHI)", color="white", fontweight="bold")

        ax = axes[1, 1]
        ax.plot(df.index, df["num_positions"], color="#66BB6A", lw=1.0)
        ax.set_title("Number of Positions", color="white", fontweight="bold")

        for ax in axes.flatten():
            for s in ax.spines.values(): s.set_edgecolor("#30363d")
            ax.tick_params(colors="white")
        fig.suptitle("Position Analysis", color="white", fontsize=14, fontweight="bold")
        plt.tight_layout()
        if save:
            p = Path(output_dir) / "position_size_distribution.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
