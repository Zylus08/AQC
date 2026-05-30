"""
aqc/portfolio/portfolio_metrics.py
====================================
Portfolio-Level Risk Metrics.

Supplements the existing ``PerformanceMetrics`` with position-level
portfolio risk measures:

* Historical VaR and Parametric VaR
* Expected Shortfall (Conditional VaR)
* Portfolio Turnover
* Concentration (Herfindahl-Hirschman Index)
* Marginal Risk Contribution

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PortfolioRiskMetrics:
    """Compute portfolio-level risk metrics.

    Parameters
    ----------
    returns:
        Portfolio return series (daily).
    weights_history:
        Optional DataFrame of portfolio weights over time
        (columns = symbols, index = dates).
    confidence_level:
        VaR / ES confidence level (default 0.95 = 95%).
    periods_per_year:
        Trading days per year (default 252).

    Examples
    --------
    >>> prm = PortfolioRiskMetrics(portfolio_returns)
    >>> var = prm.historical_var()
    >>> es = prm.expected_shortfall()
    """

    def __init__(
        self,
        returns: pd.Series,
        weights_history: Optional[pd.DataFrame] = None,
        confidence_level: float = 0.95,
        periods_per_year: int = 252,
    ) -> None:
        self.returns = returns.dropna()
        self.weights_history = weights_history
        self.confidence_level = confidence_level
        self.periods_per_year = periods_per_year

    def compute_all(self) -> dict:
        """Compute all portfolio risk metrics.

        Returns
        -------
        dict
            All metrics keyed by name.
        """
        if len(self.returns) < 2:
            return {}

        return {
            "portfolio_volatility": self.portfolio_volatility(),
            "historical_var": self.historical_var(),
            "parametric_var": self.parametric_var(),
            "expected_shortfall": self.expected_shortfall(),
            "portfolio_turnover": self.portfolio_turnover(),
            "concentration_hhi": self.concentration_hhi(),
            "max_1d_loss": self.max_1d_loss(),
            "skewness": self.skewness(),
            "kurtosis": self.kurtosis(),
        }

    # ------------------------------------------------------------------
    # Volatility
    # ------------------------------------------------------------------

    def portfolio_volatility(self) -> float:
        """Annualised portfolio volatility."""
        if len(self.returns) < 2:
            return 0.0
        return round(float(self.returns.std() * math.sqrt(self.periods_per_year)), 6)

    # ------------------------------------------------------------------
    # Value-at-Risk
    # ------------------------------------------------------------------

    def historical_var(self, horizon: int = 1) -> float:
        """Historical simulation VaR.

        Parameters
        ----------
        horizon:
            Holding period in days (default 1).

        Returns
        -------
        float
            VaR as a negative number (loss).
        """
        if len(self.returns) < 10:
            return 0.0

        alpha = 1.0 - self.confidence_level
        if horizon > 1:
            # Multi-day returns via rolling sum
            multi = self.returns.rolling(horizon).sum().dropna()
            if len(multi) < 10:
                return 0.0
            var = float(np.percentile(multi, alpha * 100))
        else:
            var = float(np.percentile(self.returns, alpha * 100))

        return round(var, 6)

    def parametric_var(self, horizon: int = 1) -> float:
        """Parametric (Gaussian) VaR.

        Assumes normally distributed returns.

        Parameters
        ----------
        horizon:
            Holding period in days.

        Returns
        -------
        float
            VaR as a negative number.
        """
        if len(self.returns) < 10:
            return 0.0

        from scipy import stats

        z = stats.norm.ppf(1.0 - self.confidence_level)
        mu = float(self.returns.mean())
        sigma = float(self.returns.std())

        var = mu * horizon + z * sigma * math.sqrt(horizon)
        return round(var, 6)

    # ------------------------------------------------------------------
    # Expected Shortfall
    # ------------------------------------------------------------------

    def expected_shortfall(self, horizon: int = 1) -> float:
        """Expected Shortfall (Conditional VaR).

        Average loss beyond the VaR threshold.

        Parameters
        ----------
        horizon:
            Holding period in days.

        Returns
        -------
        float
            ES as a negative number.
        """
        if len(self.returns) < 10:
            return 0.0

        var = self.historical_var(horizon)

        if horizon > 1:
            data = self.returns.rolling(horizon).sum().dropna()
        else:
            data = self.returns

        tail = data[data <= var]
        if len(tail) == 0:
            return var

        return round(float(tail.mean()), 6)

    # ------------------------------------------------------------------
    # Turnover
    # ------------------------------------------------------------------

    def portfolio_turnover(self) -> float:
        """Annualised portfolio turnover.

        Computed from weights history if available.
        Turnover = mean of daily sum of |weight changes|, annualised.

        Returns
        -------
        float
            Annualised turnover (e.g. 2.0 = 200% per year).
        """
        if self.weights_history is None or len(self.weights_history) < 2:
            return float("nan")

        delta = self.weights_history.diff().abs()
        daily_turnover = delta.sum(axis=1).iloc[1:]  # skip first NaN row

        if len(daily_turnover) == 0:
            return 0.0

        return round(float(daily_turnover.mean() * self.periods_per_year), 4)

    # ------------------------------------------------------------------
    # Concentration
    # ------------------------------------------------------------------

    def concentration_hhi(self, weights: Optional[dict[str, float]] = None) -> float:
        """Herfindahl-Hirschman Index of portfolio concentration.

        HHI = Σ(w_i^2), ranges from 1/N (fully diversified) to 1.0 (single asset).

        Parameters
        ----------
        weights:
            Current portfolio weights. If None, uses last row of
            weights_history (or returns NaN).

        Returns
        -------
        float
        """
        if weights is not None:
            vals = list(weights.values())
        elif self.weights_history is not None and len(self.weights_history) > 0:
            vals = self.weights_history.iloc[-1].values.tolist()
        else:
            return float("nan")

        total = sum(abs(v) for v in vals)
        if total < 1e-10:
            return float("nan")

        normalised = [abs(v) / total for v in vals]
        return round(float(sum(w ** 2 for w in normalised)), 6)

    # ------------------------------------------------------------------
    # Distribution metrics
    # ------------------------------------------------------------------

    def max_1d_loss(self) -> float:
        """Maximum single-day loss."""
        if len(self.returns) == 0:
            return 0.0
        return round(float(self.returns.min()), 6)

    def skewness(self) -> float:
        """Return distribution skewness."""
        if len(self.returns) < 3:
            return 0.0
        return round(float(self.returns.skew()), 4)

    def kurtosis(self) -> float:
        """Return distribution excess kurtosis."""
        if len(self.returns) < 4:
            return 0.0
        return round(float(self.returns.kurtosis()), 4)

    # ------------------------------------------------------------------
    # Risk contribution
    # ------------------------------------------------------------------

    def marginal_risk_contribution(
        self,
        asset_returns: pd.DataFrame,
        weights: dict[str, float],
    ) -> dict[str, float]:
        """Compute marginal risk contribution per asset.

        MRC_i = w_i * (Σ * w)_i / σ_p

        Parameters
        ----------
        asset_returns:
            Multi-column returns DataFrame (columns = symbols).
        weights:
            Current portfolio weights.

        Returns
        -------
        dict[str, float]
            Risk contribution per symbol.
        """
        symbols = [s for s in weights if s in asset_returns.columns]
        if len(symbols) < 2:
            return {s: 1.0 for s in symbols}

        ret = asset_returns[symbols].dropna()
        if len(ret) < 10:
            return {s: 1.0 / len(symbols) for s in symbols}

        cov = ret.cov().values
        w = np.array([weights[s] for s in symbols])

        port_var = float(w @ cov @ w)
        port_vol = math.sqrt(max(port_var, 1e-10))

        mrc = (cov @ w) * w / port_vol

        total_mrc = sum(abs(m) for m in mrc)
        if total_mrc < 1e-10:
            return {s: 1.0 / len(symbols) for s in symbols}

        return {
            s: round(float(abs(mrc[i]) / total_mrc), 6)
            for i, s in enumerate(symbols)
        }
