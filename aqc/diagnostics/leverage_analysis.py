"""
aqc/diagnostics/leverage_analysis.py
======================================
Leverage forensics: did leverage cause performance or drawdowns?

Author: Saksham Mishra — AlgoQuant Club
"""
from __future__ import annotations
import logging, math
from dataclasses import dataclass
from typing import Optional
import numpy as np, pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class LeverageStats:
    avg_gross: float = 0.0
    max_gross: float = 0.0
    avg_net: float = 0.0
    max_net: float = 0.0
    min_net: float = 0.0
    pct_leveraged: float = 0.0      # % bars with gross > 1
    leverage_utilisation: float = 0.0 # avg gross / max allowed

class LeverageAnalyzer:
    """Analyse portfolio leverage behaviour over time.

    Parameters
    ----------
    equity_series : pd.Series   equity per bar
    position_values : pd.DataFrame  columns=symbols, values=market_value per bar
    initial_capital : float
    max_leverage : float  max allowed (for utilisation calc)
    """
    def __init__(self, equity_series: pd.Series,
                 position_values: pd.DataFrame,
                 initial_capital: float = 100_000.0,
                 max_leverage: float = 3.0) -> None:
        self.equity = equity_series
        self.pos_values = position_values
        self.initial_capital = initial_capital
        self.max_leverage = max_leverage
        self._series: Optional[pd.DataFrame] = None

    def compute(self) -> pd.DataFrame:
        eq = self.equity.reindex(self.pos_values.index).ffill().bfill()
        eq = eq.replace(0, np.nan).ffill()
        gross = self.pos_values.abs().sum(axis=1)
        net = self.pos_values.sum(axis=1)
        df = pd.DataFrame({
            "gross_leverage": gross / eq,
            "net_leverage": net / eq,
            "gross_exposure": gross,
            "net_exposure": net,
            "equity": eq,
        }, index=self.pos_values.index)
        self._series = df
        return df

    def stats(self) -> LeverageStats:
        df = self._series if self._series is not None else self.compute()
        gl = df["gross_leverage"].dropna()
        nl = df["net_leverage"].dropna()
        return LeverageStats(
            avg_gross=round(float(gl.mean()), 4),
            max_gross=round(float(gl.max()), 4),
            avg_net=round(float(nl.mean()), 4),
            max_net=round(float(nl.max()), 4),
            min_net=round(float(nl.min()), 4),
            pct_leveraged=round(float((gl > 1.0).mean()), 4),
            leverage_utilisation=round(float(gl.mean() / self.max_leverage), 4) if self.max_leverage > 0 else 0.0,
        )

    def leverage_during_drawdowns(self, threshold: float = -0.02) -> pd.DataFrame:
        df = self._series if self._series is not None else self.compute()
        eq = df["equity"]
        peak = eq.cummax()
        dd = (eq - peak) / peak
        in_dd = dd < threshold
        return df[in_dd][["gross_leverage", "net_leverage"]].describe()

    def leverage_by_regime(self, regime_series: pd.Series) -> pd.DataFrame:
        df = self._series if self._series is not None else self.compute()
        merged = df.join(regime_series.rename("regime"), how="inner")
        return merged.groupby("regime")[["gross_leverage", "net_leverage"]].agg(["mean", "max", "std"])

    def save_report(self, path: str = "reports/leverage_report.csv") -> None:
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
        ax.plot(df.index, df["gross_leverage"], color="#4FC3F7", lw=1.2, label="Gross")
        ax.plot(df.index, df["net_leverage"], color="#FF7043", lw=1.0, alpha=0.7, label="Net")
        ax.axhline(1, color="#888", ls="--", lw=0.8)
        ax.set_title("Leverage Over Time", color="white", fontweight="bold")
        ax.legend(fontsize=9, facecolor="#21262d", labelcolor="white")

        ax = axes[0, 1]
        ax.hist(df["gross_leverage"].dropna(), bins=40, color="#4FC3F7", alpha=0.8, edgecolor="#21262d")
        ax.axvline(1, color="white", ls="--", lw=1)
        ax.set_title("Leverage Distribution", color="white", fontweight="bold")

        ax = axes[1, 0]
        eq = df["equity"]
        dd = (eq - eq.cummax()) / eq.cummax() * 100
        ax.fill_between(df.index, dd, 0, color="#EF5350", alpha=0.3)
        ax2 = ax.twinx()
        ax2.plot(df.index, df["gross_leverage"], color="#4FC3F7", lw=0.8, alpha=0.6)
        ax2.tick_params(colors="white")
        ax.set_title("Leverage During Drawdowns", color="white", fontweight="bold")

        ax = axes[1, 1]
        rolling = df["gross_leverage"].rolling(21).mean()
        ax.plot(df.index, rolling, color="#66BB6A", lw=1.2, label="21-day MA")
        ax.set_title("Rolling Average Leverage", color="white", fontweight="bold")
        ax.legend(fontsize=9, facecolor="#21262d", labelcolor="white")

        for ax in axes.flatten():
            for s in ax.spines.values(): s.set_edgecolor("#30363d")
            ax.tick_params(colors="white")
        fig.suptitle("Leverage Analysis", color="white", fontsize=14, fontweight="bold")
        plt.tight_layout()
        if save:
            p = Path(output_dir) / "leverage_over_time.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
