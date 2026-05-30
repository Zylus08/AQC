"""
aqc/diagnostics/exposure_analysis.py
======================================
Exposure forensics: was the portfolio always fully invested? Did exposure spike?

Author: Saksham Mishra — AlgoQuant Club
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Optional
import numpy as np, pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class ExposureStats:
    avg_gross: float = 0.0
    max_gross: float = 0.0
    avg_net: float = 0.0
    avg_long: float = 0.0
    avg_short: float = 0.0
    pct_fully_invested: float = 0.0
    max_long_exposure: float = 0.0
    max_short_exposure: float = 0.0

class ExposureAnalyzer:
    """Analyse portfolio exposure behaviour.

    Parameters
    ----------
    equity_series : pd.Series
    position_values : pd.DataFrame  cols=symbols, vals=signed market value
    """
    def __init__(self, equity_series: pd.Series,
                 position_values: pd.DataFrame) -> None:
        self.equity = equity_series
        self.pos_values = position_values
        self._series: Optional[pd.DataFrame] = None

    def compute(self) -> pd.DataFrame:
        eq = self.equity.reindex(self.pos_values.index).ffill().bfill()
        eq = eq.replace(0, np.nan).ffill()
        long_exp = self.pos_values.clip(lower=0).sum(axis=1) / eq
        short_exp = self.pos_values.clip(upper=0).abs().sum(axis=1) / eq
        gross = long_exp + short_exp
        net = long_exp - short_exp
        df = pd.DataFrame({
            "long_exposure": long_exp,
            "short_exposure": short_exp,
            "gross_exposure": gross,
            "net_exposure": net,
        }, index=self.pos_values.index)
        self._series = df
        return df

    def stats(self) -> ExposureStats:
        df = self._series if self._series is not None else self.compute()
        return ExposureStats(
            avg_gross=round(float(df["gross_exposure"].mean()), 4),
            max_gross=round(float(df["gross_exposure"].max()), 4),
            avg_net=round(float(df["net_exposure"].mean()), 4),
            avg_long=round(float(df["long_exposure"].mean()), 4),
            avg_short=round(float(df["short_exposure"].mean()), 4),
            pct_fully_invested=round(float((df["gross_exposure"] > 0.95).mean()), 4),
            max_long_exposure=round(float(df["long_exposure"].max()), 4),
            max_short_exposure=round(float(df["short_exposure"].max()), 4),
        )

    def exposure_by_regime(self, regime_series: pd.Series) -> pd.DataFrame:
        df = self._series if self._series is not None else self.compute()
        merged = df.join(regime_series.rename("regime"), how="inner")
        return merged.groupby("regime").mean()

    def save_report(self, path: str = "reports/exposure_report.csv") -> None:
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
        ax.plot(df.index, df["gross_exposure"], color="#4FC3F7", lw=1.2, label="Gross")
        ax.plot(df.index, df["net_exposure"], color="#FF7043", lw=1.0, alpha=0.7, label="Net")
        ax.set_title("Exposure Over Time", color="white", fontweight="bold")
        ax.legend(fontsize=9, facecolor="#21262d", labelcolor="white")

        ax = axes[0, 1]
        ax.fill_between(df.index, df["long_exposure"], color="#66BB6A", alpha=0.6, label="Long")
        ax.fill_between(df.index, -df["short_exposure"], color="#EF5350", alpha=0.6, label="Short")
        ax.set_title("Long vs Short Exposure", color="white", fontweight="bold")
        ax.legend(fontsize=9, facecolor="#21262d", labelcolor="white")

        ax = axes[1, 0]
        ax.hist(df["gross_exposure"].dropna(), bins=40, color="#4FC3F7", alpha=0.8, edgecolor="#21262d")
        ax.set_title("Exposure Distribution", color="white", fontweight="bold")

        ax = axes[1, 1]
        r21 = df["gross_exposure"].rolling(21).mean()
        ax.plot(df.index, r21, color="#AB47BC", lw=1.2, label="21d MA Gross")
        ax.set_title("Rolling Exposure", color="white", fontweight="bold")
        ax.legend(fontsize=9, facecolor="#21262d", labelcolor="white")

        for ax in axes.flatten():
            for s in ax.spines.values(): s.set_edgecolor("#30363d")
            ax.tick_params(colors="white")
        fig.suptitle("Exposure Analysis", color="white", fontsize=14, fontweight="bold")
        plt.tight_layout()
        if save:
            p = Path(output_dir) / "exposure_over_time.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
