"""
aqc/volatility/forecasting_engine.py
=====================================
Multi-Model Volatility Forecasting Engine.

Orchestrates multiple volatility estimators (EWMA, GARCH, historical)
and produces:

* Next-period volatility forecasts
* Confidence intervals
* Volatility regime labels (Low / Normal / High / Extreme)
* Ensemble forecasts (weighted average of models)

The engine is designed for integration with the backtest loop: call
``update()`` on each bar and ``forecast()`` to get the next-period
estimate.

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from aqc.volatility.ewma import ewma_volatility, ewma_variance
from aqc.volatility.garch import GARCH11, GARCHResult

logger = logging.getLogger(__name__)


class VolRegime(Enum):
    """Volatility regime classification."""

    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


@dataclass
class ForecastResult:
    """Container for a volatility forecast.

    Attributes
    ----------
    forecast_vol:
        Point forecast of annualised volatility.
    ci_lower:
        Lower 95% confidence interval bound.
    ci_upper:
        Upper 95% confidence interval bound.
    regime:
        Current volatility regime label.
    model_vols:
        Per-model volatility estimates: ``{"ewma": ..., "garch": ..., "hist": ...}``.
    realized_vol:
        Most recent realized volatility (for comparison).
    timestamp:
        Forecast timestamp.
    """

    forecast_vol: float = 0.0
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    regime: VolRegime = VolRegime.NORMAL
    model_vols: dict = field(default_factory=dict)
    realized_vol: float = 0.0
    timestamp: Optional[pd.Timestamp] = None


class VolatilityForecastEngine:
    """Multi-model volatility forecasting engine.

    Combines EWMA, GARCH(1,1), and historical volatility into an
    ensemble forecast with regime detection.

    Parameters
    ----------
    ewma_decay:
        Decay factor for EWMA model (default 0.94).
    hist_window:
        Rolling window for historical volatility (default 21).
    garch_refit_every:
        Re-fit GARCH every N observations (default 60). Set to 0 to disable.
    ann_factor:
        Annualisation factor (default 252 for daily).
    weights:
        Model weights for ensemble: ``(w_ewma, w_garch, w_hist)``.
        Normalised automatically. Default ``(0.4, 0.35, 0.25)``.
    regime_thresholds:
        Percentile thresholds for regime classification.
        Default: ``{"low": 25, "high": 75, "extreme": 95}``.

    Examples
    --------
    >>> engine = VolatilityForecastEngine()
    >>> result = engine.fit_and_forecast(close_prices)
    """

    def __init__(
        self,
        ewma_decay: float = 0.94,
        hist_window: int = 21,
        garch_refit_every: int = 60,
        ann_factor: int = 252,
        weights: tuple[float, float, float] = (0.4, 0.35, 0.25),
        regime_thresholds: Optional[dict] = None,
    ) -> None:
        self.ewma_decay = ewma_decay
        self.hist_window = hist_window
        self.garch_refit_every = garch_refit_every
        self.ann_factor = ann_factor

        # Normalise weights
        total = sum(weights)
        self.w_ewma = weights[0] / total
        self.w_garch = weights[1] / total
        self.w_hist = weights[2] / total

        self.regime_thresholds = regime_thresholds or {
            "low": 25, "high": 75, "extreme": 95,
        }

        # Internal state
        self._garch_model = GARCH11(ann_factor=ann_factor)
        self._garch_result: Optional[GARCHResult] = None
        self._returns: Optional[pd.Series] = None
        self._vol_history: list[float] = []

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def fit_and_forecast(
        self,
        prices: pd.Series,
        horizon: int = 1,
    ) -> ForecastResult:
        """Fit all models and produce an ensemble forecast.

        This is the primary high-level API. Pass a close price series
        and get back a comprehensive forecast result.

        Parameters
        ----------
        prices:
            Close price series (must have at least 30 observations).
        horizon:
            Forecast horizon in periods (default 1 = next bar).

        Returns
        -------
        ForecastResult
            Ensemble volatility forecast with confidence intervals and regime.
        """
        returns = np.log(prices / prices.shift(1)).dropna()
        self._returns = returns

        if len(returns) < 30:
            logger.warning("Insufficient data for vol forecast (%d bars)", len(returns))
            return ForecastResult()

        # --- EWMA ---
        ewma_vol = self._compute_ewma(returns)

        # --- Historical ---
        hist_vol = self._compute_historical(returns)

        # --- GARCH ---
        garch_vol = self._compute_garch(returns)

        # --- Ensemble ---
        ensemble_vol = (
            self.w_ewma * ewma_vol
            + self.w_garch * garch_vol
            + self.w_hist * hist_vol
        )

        # --- Confidence intervals ---
        ci_lower, ci_upper = self._compute_ci(
            ensemble_vol, ewma_vol, garch_vol, hist_vol
        )

        # --- Regime ---
        self._vol_history.append(ensemble_vol)
        regime = self._classify_regime(ensemble_vol)

        # --- Realized vol ---
        realized = hist_vol

        ts = prices.index[-1] if hasattr(prices.index, "__getitem__") else None

        return ForecastResult(
            forecast_vol=round(ensemble_vol, 6),
            ci_lower=round(ci_lower, 6),
            ci_upper=round(ci_upper, 6),
            regime=regime,
            model_vols={
                "ewma": round(ewma_vol, 6),
                "garch": round(garch_vol, 6),
                "historical": round(hist_vol, 6),
            },
            realized_vol=round(realized, 6),
            timestamp=ts,
        )

    def compute_full_series(
        self,
        prices: pd.Series,
    ) -> pd.DataFrame:
        """Compute full time series of volatility estimates from all models.

        Parameters
        ----------
        prices:
            Close price series.

        Returns
        -------
        pd.DataFrame
            Columns: ``ewma_vol``, ``garch_vol``, ``hist_vol``, ``ensemble_vol``,
            ``regime``.
        """
        returns = np.log(prices / prices.shift(1)).dropna()

        # EWMA
        ewma_vol = ewma_volatility(
            returns, decay=self.ewma_decay, annualise=True, ann_factor=self.ann_factor,
        )

        # Historical
        hist_vol = returns.rolling(
            window=self.hist_window, min_periods=self.hist_window
        ).std() * np.sqrt(self.ann_factor)

        # GARCH
        garch_vol = self._fit_garch_series(returns)

        # Build DataFrame
        df = pd.DataFrame({
            "ewma_vol": ewma_vol,
            "garch_vol": garch_vol,
            "hist_vol": hist_vol,
        }, index=returns.index)

        df["ensemble_vol"] = (
            self.w_ewma * df["ewma_vol"]
            + self.w_garch * df["garch_vol"]
            + self.w_hist * df["hist_vol"]
        )

        # Regime labels
        df["regime"] = df["ensemble_vol"].apply(
            lambda v: self._classify_regime(v).value if not np.isnan(v) else "NORMAL"
        )

        return df.dropna()

    def generate_report(
        self,
        prices: pd.Series,
        output_path: str = "reports/volatility_report.csv",
    ) -> pd.DataFrame:
        """Generate a full volatility report and save to CSV.

        Parameters
        ----------
        prices:
            Close price series.
        output_path:
            Output CSV path.

        Returns
        -------
        pd.DataFrame
            Full volatility report.
        """
        from pathlib import Path

        df = self.compute_full_series(prices)

        # Add realized vol (shifted for forecast comparison)
        returns = np.log(prices / prices.shift(1)).dropna()
        df["realized_1d"] = returns.abs() * np.sqrt(self.ann_factor)
        df["forecast_error"] = df["ensemble_vol"] - df["realized_1d"]

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path)
        logger.info("Volatility report saved to %s (%d rows)", output_path, len(df))

        return df

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_forecast_vs_realized(
        self, df: pd.DataFrame, save: bool = True, output_dir: str = "reports",
    ) -> None:
        """Plot forecast volatility vs realized with regime shading.

        Parameters
        ----------
        df:
            Output from ``compute_full_series()`` or ``generate_report()``.
        save:
            If True, save to file.
        output_dir:
            Directory for output file.
        """
        import matplotlib.pyplot as plt
        from pathlib import Path

        plt.style.use("dark_background")
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(14, 9), facecolor="#0d1117", height_ratios=[3, 1],
            sharex=True,
        )

        for ax in (ax1, ax2):
            ax.set_facecolor("#161b22")

        # Top panel: vol forecast vs realized
        ax1.plot(df.index, df["ensemble_vol"] * 100, color="#4FC3F7",
                 linewidth=1.5, label="Ensemble Forecast", alpha=0.9)
        ax1.plot(df.index, df["ewma_vol"] * 100, color="#81C784",
                 linewidth=0.8, alpha=0.5, label="EWMA")
        ax1.plot(df.index, df["garch_vol"] * 100, color="#FFB74D",
                 linewidth=0.8, alpha=0.5, label="GARCH")
        ax1.plot(df.index, df["hist_vol"] * 100, color="#F06292",
                 linewidth=0.8, alpha=0.5, label="Historical")

        if "realized_1d" in df.columns:
            ax1.scatter(df.index, df["realized_1d"] * 100, color="white",
                        s=1, alpha=0.15, label="Realized (1d)")

        ax1.set_title("Volatility Forecast vs Realized",
                       color="white", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Annualised Volatility (%)", color="white", fontsize=11)
        ax1.legend(fontsize=9, facecolor="#21262d", labelcolor="white", loc="upper left")
        ax1.tick_params(colors="white")

        # Regime shading
        regime_colors = {"LOW": "#1B5E20", "NORMAL": "#1565C0", "HIGH": "#E65100", "EXTREME": "#B71C1C"}
        if "regime" in df.columns:
            for regime, color in regime_colors.items():
                mask = df["regime"] == regime
                if mask.any():
                    ax1.fill_between(df.index, 0, ax1.get_ylim()[1],
                                     where=mask, alpha=0.08, color=color, label=f"_{regime}")

        # Bottom panel: regime timeline
        regime_map = {"LOW": 0, "NORMAL": 1, "HIGH": 2, "EXTREME": 3}
        if "regime" in df.columns:
            regime_numeric = df["regime"].map(regime_map).fillna(1)
            colors = [regime_colors.get(r, "#1565C0") for r in df["regime"]]
            ax2.bar(df.index, regime_numeric, color=colors, width=1.5, alpha=0.8)
            ax2.set_yticks([0, 1, 2, 3])
            ax2.set_yticklabels(["Low", "Normal", "High", "Extreme"],
                                color="white", fontsize=9)
            ax2.set_ylabel("Regime", color="white", fontsize=10)

        for ax in (ax1, ax2):
            for spine in ax.spines.values():
                spine.set_edgecolor("#30363d")
            ax.tick_params(colors="white")

        plt.tight_layout()

        if save:
            path = Path(output_dir) / "vol_forecast_vs_realized.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
            logger.info("Plot saved to %s", path)

        plt.close(fig)

    def plot_volatility_clusters(
        self, df: pd.DataFrame, save: bool = True, output_dir: str = "reports",
    ) -> None:
        """Plot volatility clustering (autocorrelation of squared returns).

        Parameters
        ----------
        df:
            Volatility DataFrame with ``ensemble_vol``.
        save / output_dir:
            File save options.
        """
        import matplotlib.pyplot as plt
        from pathlib import Path

        plt.style.use("dark_background")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0d1117")

        for ax in axes:
            ax.set_facecolor("#161b22")

        # Left: rolling vol with cluster highlights
        vol = df["ensemble_vol"] * 100
        mean_vol = vol.mean()
        ax = axes[0]
        ax.plot(df.index, vol, color="#4FC3F7", linewidth=1.2)
        ax.axhline(mean_vol, color="#888", linestyle="--", linewidth=0.8)
        high_vol = vol > vol.quantile(0.75)
        ax.fill_between(df.index, vol, mean_vol, where=high_vol,
                         color="#FF5252", alpha=0.3, label="High vol cluster")
        ax.set_title("Volatility Clusters", color="white", fontsize=12, fontweight="bold")
        ax.set_ylabel("Vol (%)", color="white")
        ax.legend(fontsize=9, facecolor="#21262d", labelcolor="white")
        ax.tick_params(colors="white")

        # Right: vol distribution
        ax = axes[1]
        ax.hist(vol.dropna(), bins=30, color="#4FC3F7", edgecolor="#21262d", alpha=0.8)
        ax.axvline(mean_vol, color="white", linestyle="--", linewidth=1.5,
                    label=f"Mean: {mean_vol:.1f}%")
        ax.axvline(vol.quantile(0.95), color="#FF5252", linestyle="--", linewidth=1,
                    label=f"95th: {vol.quantile(0.95):.1f}%")
        ax.set_title("Volatility Distribution", color="white", fontsize=12, fontweight="bold")
        ax.set_xlabel("Vol (%)", color="white")
        ax.legend(fontsize=9, facecolor="#21262d", labelcolor="white")
        ax.tick_params(colors="white")

        for ax in axes:
            for spine in ax.spines.values():
                spine.set_edgecolor("#30363d")

        plt.tight_layout()

        if save:
            path = Path(output_dir) / "vol_clusters.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
            logger.info("Plot saved to %s", path)

        plt.close(fig)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_ewma(self, returns: pd.Series) -> float:
        """Compute latest EWMA volatility."""
        vol_series = ewma_volatility(
            returns, decay=self.ewma_decay, annualise=True, ann_factor=self.ann_factor,
        )
        valid = vol_series.dropna()
        return float(valid.iloc[-1]) if len(valid) > 0 else 0.0

    def _compute_historical(self, returns: pd.Series) -> float:
        """Compute latest historical volatility."""
        if len(returns) < self.hist_window:
            return 0.0
        return float(returns.iloc[-self.hist_window:].std() * np.sqrt(self.ann_factor))

    def _compute_garch(self, returns: pd.Series) -> float:
        """Compute latest GARCH volatility (fits if needed)."""
        try:
            need_fit = (
                self._garch_result is None
                or not self._garch_result.converged
                or (self.garch_refit_every > 0
                    and len(returns) % self.garch_refit_every == 0)
            )

            if need_fit:
                self._garch_result = self._garch_model.fit(returns)

            if self._garch_result.converged and self._garch_result.conditional_variance is not None:
                last_var = float(self._garch_result.conditional_variance.iloc[-1])
                return float(np.sqrt(last_var * self.ann_factor))
            else:
                return self._compute_historical(returns)
        except Exception as e:
            logger.debug("GARCH fitting failed: %s — falling back to historical", e)
            return self._compute_historical(returns)

    def _fit_garch_series(self, returns: pd.Series) -> pd.Series:
        """Fit GARCH and return full conditional volatility series."""
        try:
            result = self._garch_model.fit(returns)
            if result.converged and result.conditional_variance is not None:
                self._garch_result = result
                return self._garch_model.conditional_volatility(result, annualise=True)
        except Exception as e:
            logger.debug("GARCH series failed: %s", e)

        # Fallback: use EWMA
        return ewma_volatility(
            returns, decay=self.ewma_decay, annualise=True, ann_factor=self.ann_factor,
        )

    def _compute_ci(
        self,
        ensemble: float,
        ewma: float,
        garch: float,
        hist: float,
    ) -> tuple[float, float]:
        """Compute 95% confidence intervals from model disagreement.

        Uses the spread between models as a proxy for forecast uncertainty.
        """
        model_vols = [ewma, garch, hist]
        model_std = float(np.std(model_vols))

        # If models agree well, use a minimum CI width
        min_ci_width = ensemble * 0.15  # at least 15% of point estimate
        ci_width = max(model_std * 1.96, min_ci_width)

        ci_lower = max(0.0, ensemble - ci_width)
        ci_upper = ensemble + ci_width

        return ci_lower, ci_upper

    def _classify_regime(self, vol: float) -> VolRegime:
        """Classify current vol level into a regime.

        Uses the running history of vol estimates and percentile thresholds.
        """
        if len(self._vol_history) < 10:
            return VolRegime.NORMAL

        arr = np.array(self._vol_history)
        p_low = np.percentile(arr, self.regime_thresholds["low"])
        p_high = np.percentile(arr, self.regime_thresholds["high"])
        p_extreme = np.percentile(arr, self.regime_thresholds["extreme"])

        if vol >= p_extreme:
            return VolRegime.EXTREME
        elif vol >= p_high:
            return VolRegime.HIGH
        elif vol <= p_low:
            return VolRegime.LOW
        else:
            return VolRegime.NORMAL
