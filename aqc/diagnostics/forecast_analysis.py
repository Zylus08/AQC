"""
aqc/diagnostics/forecast_analysis.py
======================================
Volatility forecast validation: are forecasts accurate? Which model is best?

Author: Saksham Mishra — AlgoQuant Club
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Optional
import numpy as np, pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class ForecastAccuracy:
    model: str = ""
    mae: float = 0.0
    rmse: float = 0.0
    mape: float = 0.0
    bias: float = 0.0
    correlation: float = 0.0
    n_obs: int = 0

class ForecastAnalyzer:
    """Validate volatility forecast accuracy against realised vol.

    Parameters
    ----------
    vol_data : pd.DataFrame
        Must contain columns for model forecasts and realised vol.
        Expected cols: ewma_vol, garch_vol, hist_vol, ensemble_vol, realized_1d
    """
    def __init__(self, vol_data: pd.DataFrame) -> None:
        self.vol_data = vol_data.dropna(how="all")
        self._models = ["ewma_vol", "garch_vol", "hist_vol", "ensemble_vol"]

    def compute_accuracy(self) -> list[ForecastAccuracy]:
        results = []
        realized_col = self._find_realized_col()
        if realized_col is None:
            return results
        realized = self.vol_data[realized_col].dropna()
        for model in self._models:
            if model not in self.vol_data.columns:
                continue
            forecast = self.vol_data[model].dropna()
            common = forecast.index.intersection(realized.index)
            if len(common) < 10:
                continue
            f = forecast.loc[common].values
            r = realized.loc[common].values
            err = f - r
            abs_err = np.abs(err)
            results.append(ForecastAccuracy(
                model=model,
                mae=round(float(np.mean(abs_err)), 6),
                rmse=round(float(np.sqrt(np.mean(err ** 2))), 6),
                mape=round(float(np.mean(abs_err / np.clip(r, 1e-8, None)) * 100), 2),
                bias=round(float(np.mean(err)), 6),
                correlation=round(float(np.corrcoef(f, r)[0, 1]), 4),
                n_obs=len(common),
            ))
        return results

    def accuracy_table(self) -> pd.DataFrame:
        accs = self.compute_accuracy()
        if not accs:
            return pd.DataFrame()
        return pd.DataFrame([a.__dict__ for a in accs]).set_index("model")

    def best_model(self) -> str:
        tbl = self.accuracy_table()
        if tbl.empty:
            return "unknown"
        return str(tbl["rmse"].idxmin())

    def error_by_regime(self, regime_series: pd.Series) -> pd.DataFrame:
        realized_col = self._find_realized_col()
        if realized_col is None or "ensemble_vol" not in self.vol_data.columns:
            return pd.DataFrame()
        err = (self.vol_data["ensemble_vol"] - self.vol_data[realized_col]).abs()
        merged = err.to_frame("abs_error").join(regime_series.rename("regime"), how="inner")
        return merged.groupby("regime")["abs_error"].agg(["mean", "std", "count"])

    def save_report(self, path: str = "reports/forecast_validation_report.csv") -> None:
        from pathlib import Path
        tbl = self.accuracy_table()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        tbl.to_csv(path)

    def plot(self, save: bool = True, output_dir: str = "reports") -> None:
        import matplotlib.pyplot as plt
        from pathlib import Path
        realized_col = self._find_realized_col()
        if realized_col is None or "ensemble_vol" not in self.vol_data.columns:
            return
        f = self.vol_data["ensemble_vol"].dropna()
        r = self.vol_data[realized_col].dropna()
        common = f.index.intersection(r.index)
        if len(common) < 10:
            return
        f, r = f.loc[common], r.loc[common]
        err = f - r

        fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor="#0d1117")
        for ax in axes.flatten(): ax.set_facecolor("#161b22")

        ax = axes[0, 0]
        ax.plot(err.index, err * 100, color="#4FC3F7", lw=0.8, alpha=0.7)
        ax.axhline(0, color="white", lw=0.8, ls="--")
        ax.fill_between(err.index, err * 100, 0, where=err > 0, color="#EF5350", alpha=0.2)
        ax.fill_between(err.index, err * 100, 0, where=err < 0, color="#66BB6A", alpha=0.2)
        ax.set_title("Forecast Error Over Time", color="white", fontweight="bold")
        ax.set_ylabel("Error (%)", color="white")

        ax = axes[0, 1]
        ax.scatter(r * 100, f * 100, s=3, alpha=0.3, color="#4FC3F7")
        lims = [min(r.min(), f.min()) * 100, max(r.max(), f.max()) * 100]
        ax.plot(lims, lims, color="white", lw=1, ls="--")
        ax.set_title("Forecast vs Realised", color="white", fontweight="bold")
        ax.set_xlabel("Realised (%)", color="white")
        ax.set_ylabel("Forecast (%)", color="white")

        ax = axes[1, 0]
        ax.hist(err * 100, bins=40, color="#AB47BC", alpha=0.8, edgecolor="#21262d")
        ax.axvline(0, color="white", lw=1, ls="--")
        ax.set_title("Error Distribution", color="white", fontweight="bold")

        ax = axes[1, 1]
        tbl = self.accuracy_table()
        if not tbl.empty:
            models = tbl.index.tolist()
            rmses = tbl["rmse"].values * 100
            c = ["#4FC3F7", "#FF7043", "#66BB6A", "#AB47BC"]
            ax.barh(models, rmses, color=c[:len(models)], alpha=0.85)
            ax.set_title("RMSE by Model (%)", color="white", fontweight="bold")

        for ax in axes.flatten():
            for s in ax.spines.values(): s.set_edgecolor("#30363d")
            ax.tick_params(colors="white")
        fig.suptitle("Forecast Validation", color="white", fontsize=14, fontweight="bold")
        plt.tight_layout()
        if save:
            p = Path(output_dir) / "forecast_error_distribution.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

    def _find_realized_col(self) -> Optional[str]:
        for c in ["realized_1d", "realised_vol", "realized_vol", "hist_vol"]:
            if c in self.vol_data.columns:
                return c
        return None
