"""
aqc/diagnostics/regime_analysis.py
====================================
Regime-specific performance forensics: which regimes generate/destroy alpha?

Author: Saksham Mishra — AlgoQuant Club
"""
from __future__ import annotations
import logging, math
from typing import Optional
import numpy as np, pandas as pd

logger = logging.getLogger(__name__)

class RegimePerformanceAnalyzer:
    """Break down portfolio performance by regime.

    Parameters
    ----------
    returns : pd.Series          daily portfolio returns
    regime_data : pd.DataFrame   columns include vol_regime, trend_regime
    ann_factor : int
    """
    def __init__(self, returns: pd.Series, regime_data: pd.DataFrame,
                 ann_factor: int = 252) -> None:
        self.returns = returns
        self.regime_data = regime_data
        self.ann_factor = ann_factor

    def by_vol_regime(self) -> pd.DataFrame:
        return self._compute_by("vol_regime")

    def by_trend_regime(self) -> pd.DataFrame:
        return self._compute_by("trend_regime")

    def _compute_by(self, col: str) -> pd.DataFrame:
        if col not in self.regime_data.columns:
            return pd.DataFrame()
        merged = self.returns.to_frame("ret").join(
            self.regime_data[col].rename("regime"), how="inner")
        rows = []
        for regime, grp in merged.groupby("regime"):
            r = grp["ret"]
            if len(r) < 2:
                continue
            std = float(r.std())
            mean = float(r.mean())
            sharpe = (mean / std * math.sqrt(self.ann_factor)) if std > 1e-10 else np.nan
            ds = r[r < 0]
            ds_std = float(ds.std()) if len(ds) > 1 else np.nan
            sortino = (mean / ds_std * math.sqrt(self.ann_factor)) if ds_std and ds_std > 1e-10 else np.nan
            n_years = len(r) / self.ann_factor
            cum = (1 + r).prod()
            cagr = (cum ** (1 / n_years) - 1) if n_years > 0 else 0
            eq = (1 + r).cumprod()
            dd = ((eq - eq.cummax()) / eq.cummax()).min()
            rows.append({
                "regime": regime,
                "n_bars": len(r),
                "pct_time": round(len(r) / len(self.returns), 4),
                "sharpe": round(sharpe, 4) if not np.isnan(sharpe) else np.nan,
                "sortino": round(sortino, 4) if not np.isnan(sortino) else np.nan,
                "cagr": round(cagr, 6),
                "max_drawdown": round(float(dd), 4),
                "win_rate": round(float((r > 0).mean()), 4),
                "avg_return": round(mean, 6),
                "total_return": round(float(cum - 1), 6),
                "contribution": round(float(r.sum()), 6),
            })
        return pd.DataFrame(rows).set_index("regime")

    def regime_contribution(self) -> pd.DataFrame:
        """How much each vol regime contributed to total return."""
        vol = self.by_vol_regime()
        if vol.empty:
            return vol
        total = vol["contribution"].sum()
        if abs(total) > 1e-10:
            vol["pct_contribution"] = round(vol["contribution"] / total * 100, 2)
        return vol[["n_bars", "pct_time", "contribution", "pct_contribution", "sharpe"]]

    def save_report(self, path: str = "reports/regime_performance_report.csv") -> None:
        from pathlib import Path
        vol = self.by_vol_regime()
        trend = self.by_trend_regime()
        combined = pd.concat({"vol_regime": vol, "trend_regime": trend})
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(path)

    def plot(self, save: bool = True, output_dir: str = "reports") -> None:
        import matplotlib.pyplot as plt
        from pathlib import Path
        vol = self.by_vol_regime()
        trend = self.by_trend_regime()
        if vol.empty and trend.empty:
            return

        fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="#0d1117")
        for ax in axes: ax.set_facecolor("#161b22")

        # 1. Vol regime Sharpe heatmap
        ax = axes[0]
        if not vol.empty and "sharpe" in vol.columns:
            regimes = vol.index.tolist()
            sharpes = vol["sharpe"].fillna(0).values
            colors = ["#1B5E20" if s > 0 else "#B71C1C" for s in sharpes]
            ax.barh(regimes, sharpes, color=colors, alpha=0.85)
            ax.set_title("Sharpe by Vol Regime", color="white", fontweight="bold")
            ax.axvline(0, color="white", lw=0.8)

        # 2. Trend regime Sharpe
        ax = axes[1]
        if not trend.empty and "sharpe" in trend.columns:
            regimes = trend.index.tolist()
            sharpes = trend["sharpe"].fillna(0).values
            colors = ["#1B5E20" if s > 0 else "#B71C1C" for s in sharpes]
            ax.barh(regimes, sharpes, color=colors, alpha=0.85)
            ax.set_title("Sharpe by Trend Regime", color="white", fontweight="bold")
            ax.axvline(0, color="white", lw=0.8)

        # 3. Contribution pie
        ax = axes[2]
        if not vol.empty and "contribution" in vol.columns:
            contrib = vol["contribution"]
            labels = contrib.index.tolist()
            vals = contrib.abs().values
            c_colors = ["#4FC3F7", "#FF7043", "#66BB6A", "#AB47BC"]
            ax.pie(vals, labels=labels, colors=c_colors[:len(labels)],
                   autopct="%1.1f%%", textprops={"color": "white", "fontsize": 9})
            ax.set_title("Return Contribution by Regime", color="white", fontweight="bold")

        for ax in axes[:2]:
            for s in ax.spines.values(): s.set_edgecolor("#30363d")
            ax.tick_params(colors="white")
        fig.suptitle("Regime Performance Analysis", color="white", fontsize=14, fontweight="bold")
        plt.tight_layout()
        if save:
            p = Path(output_dir) / "regime_performance_heatmap.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
