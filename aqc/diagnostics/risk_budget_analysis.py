"""
aqc/diagnostics/risk_budget_analysis.py
=========================================
Risk budget forensics: was the risk engine working correctly?

Author: Saksham Mishra — AlgoQuant Club
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Optional
import numpy as np, pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class RiskBudgetStats:
    avg_utilisation: float = 0.0
    max_utilisation: float = 0.0
    pct_over_budget: float = 0.0
    pct_under_50: float = 0.0
    avg_forecast_vol: float = 0.0
    avg_realised_vol: float = 0.0

class RiskBudgetAnalyzer:
    """Analyse risk budget utilisation: actual_risk / target_risk.

    Parameters
    ----------
    equity_series : pd.Series
    forecast_vol_series : pd.Series  per-bar forecast vol
    realised_vol_series : pd.Series  per-bar realised vol
    position_values : pd.DataFrame   signed market values
    target_vol : float               target portfolio vol
    """
    def __init__(self, equity_series: pd.Series,
                 forecast_vol_series: pd.Series,
                 realised_vol_series: pd.Series,
                 position_values: pd.DataFrame,
                 target_vol: float = 0.10) -> None:
        self.equity = equity_series
        self.forecast_vol = forecast_vol_series
        self.realised_vol = realised_vol_series
        self.pos_values = position_values
        self.target_vol = target_vol
        self._series: Optional[pd.DataFrame] = None

    def compute(self) -> pd.DataFrame:
        eq = self.equity.reindex(self.pos_values.index).ffill().bfill().replace(0, np.nan).ffill()
        gross = self.pos_values.abs().sum(axis=1)
        actual_vol = self.realised_vol.reindex(self.pos_values.index).ffill()
        leverage = gross / eq
        actual_risk = leverage * actual_vol
        target_risk = pd.Series(self.target_vol, index=self.pos_values.index)
        utilisation = actual_risk / target_risk.replace(0, np.nan)
        df = pd.DataFrame({
            "forecast_vol": self.forecast_vol.reindex(self.pos_values.index).ffill(),
            "realised_vol": actual_vol,
            "leverage": leverage,
            "actual_risk": actual_risk,
            "target_risk": target_risk,
            "utilisation": utilisation,
        }, index=self.pos_values.index)
        self._series = df
        return df

    def stats(self) -> RiskBudgetStats:
        df = self._series if self._series is not None else self.compute()
        u = df["utilisation"].dropna()
        fv = df["forecast_vol"].dropna()
        rv = df["realised_vol"].dropna()
        return RiskBudgetStats(
            avg_utilisation=round(float(u.mean()), 4),
            max_utilisation=round(float(u.max()), 4),
            pct_over_budget=round(float((u > 1.0).mean()), 4),
            pct_under_50=round(float((u < 0.5).mean()), 4),
            avg_forecast_vol=round(float(fv.mean()), 6) if len(fv) > 0 else 0.0,
            avg_realised_vol=round(float(rv.mean()), 6) if len(rv) > 0 else 0.0,
        )

    def save_report(self, path: str = "reports/risk_budget_report.csv") -> None:
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
        u = df["utilisation"].dropna()
        ax.plot(u.index, u, color="#4FC3F7", lw=1.0)
        ax.axhline(1.0, color="#EF5350", ls="--", lw=1, label="Target (100%)")
        ax.axhline(0.5, color="#FFB74D", ls="--", lw=0.8, alpha=0.5, label="50%")
        ax.fill_between(u.index, u, 1, where=u > 1, color="#EF5350", alpha=0.2)
        ax.set_title("Risk Utilisation Over Time", color="white", fontweight="bold")
        ax.legend(fontsize=8, facecolor="#21262d", labelcolor="white")

        ax = axes[0, 1]
        ax.hist(u, bins=40, color="#4FC3F7", alpha=0.8, edgecolor="#21262d")
        ax.axvline(1.0, color="#EF5350", ls="--", lw=1)
        ax.set_title("Utilisation Distribution", color="white", fontweight="bold")

        ax = axes[1, 0]
        over = (u > 1.0).rolling(21).mean() * 100
        under = (u < 0.5).rolling(21).mean() * 100
        ax.fill_between(over.index, over, color="#EF5350", alpha=0.5, label="Over-budget %")
        ax.fill_between(under.index, -under, color="#66BB6A", alpha=0.5, label="Under-budget %")
        ax.set_title("Over-risk vs Under-risk Periods", color="white", fontweight="bold")
        ax.legend(fontsize=8, facecolor="#21262d", labelcolor="white")

        ax = axes[1, 1]
        fv = df["forecast_vol"].dropna()
        rv = df["realised_vol"].dropna()
        ax.plot(fv.index, fv, color="#4FC3F7", lw=1.0, alpha=0.8, label="Forecast")
        ax.plot(rv.index, rv, color="#FF7043", lw=1.0, alpha=0.8, label="Realised")
        ax.set_title("Forecast vs Realised Vol", color="white", fontweight="bold")
        ax.legend(fontsize=8, facecolor="#21262d", labelcolor="white")

        for ax in axes.flatten():
            for s in ax.spines.values(): s.set_edgecolor("#30363d")
            ax.tick_params(colors="white")
        fig.suptitle("Risk Budget Analysis", color="white", fontsize=14, fontweight="bold")
        plt.tight_layout()
        if save:
            p = Path(output_dir) / "risk_budget_utilisation.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
