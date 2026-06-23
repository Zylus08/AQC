"""
aqc/alpha/alpha_decay_monitor.py
==================================
Alpha Decay Analysis Module.

Measures the half-life of alpha signal predictive power by computing
autocorrelation of alpha returns at increasing lag horizons.  A fast-
decaying alpha requires higher turnover and tighter execution; a slow-
decaying alpha can afford lower frequency rebalancing.

Integrates with the existing
:mod:`aqc.live_validation.alpha_decay` module for forward-looking
decay tracking.

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AlphaDecayMonitor:
    """Compute and track alpha signal decay characteristics.

    Parameters
    ----------
    max_lag:
        Maximum lag (in bars) to evaluate for autocorrelation and IC decay.
    min_observations:
        Minimum number of observations required for a meaningful estimate.

    Examples
    --------
    >>> monitor = AlphaDecayMonitor(max_lag=50)
    >>> result = monitor.analyse(predictions, actuals)
    >>> print(f"Half-life: {result['halflife_bars']:.1f} bars")
    """

    def __init__(
        self,
        max_lag: int = 50,
        min_observations: int = 100,
    ) -> None:
        self.max_lag = max_lag
        self.min_observations = min_observations

    # ------------------------------------------------------------------
    # Primary analysis
    # ------------------------------------------------------------------

    def analyse(
        self,
        predictions: pd.Series,
        actuals: pd.Series,
    ) -> dict:
        """Run full decay analysis on a prediction-actual series.

        Parameters
        ----------
        predictions:
            Alpha scores at time *t*.
        actuals:
            Realised returns at time *t*.

        Returns
        -------
        dict
            Keys: ``ic_by_lag``, ``autocorrelation``, ``halflife_bars``,
            ``decay_rate``, ``is_decaying``.
        """
        if len(predictions) < self.min_observations:
            logger.warning(
                "Insufficient data for decay analysis: %d < %d",
                len(predictions), self.min_observations,
            )
            return {
                "ic_by_lag": {},
                "autocorrelation": {},
                "halflife_bars": float("inf"),
                "decay_rate": 0.0,
                "is_decaying": False,
            }

        ic_by_lag = self.compute_ic_decay(predictions, actuals)
        autocorr = self.compute_return_autocorrelation(predictions, actuals)
        halflife = self.estimate_halflife(ic_by_lag)
        decay_rate = self._decay_rate(ic_by_lag)

        return {
            "ic_by_lag": ic_by_lag,
            "autocorrelation": autocorr,
            "halflife_bars": halflife,
            "decay_rate": decay_rate,
            "is_decaying": halflife < self.max_lag,
        }

    # ------------------------------------------------------------------
    # IC decay curve
    # ------------------------------------------------------------------

    def compute_ic_decay(
        self,
        predictions: pd.Series,
        actuals: pd.Series,
    ) -> dict[int, float]:
        """Compute information coefficient at each lag.

        At lag *k*, the IC is the rank correlation between
        ``predictions[t]`` and ``actuals[t + k]``.

        Parameters
        ----------
        predictions:
            Alpha scores.
        actuals:
            Realised returns.

        Returns
        -------
        dict[int, float]
            ``{lag: IC}``
        """
        ic_by_lag: dict[int, float] = {}

        for lag in range(0, self.max_lag + 1):
            shifted_actuals = actuals.shift(-lag)
            aligned = pd.DataFrame({
                "pred": predictions,
                "actual": shifted_actuals,
            }).dropna()

            if len(aligned) < 20:
                break

            ic = float(aligned["pred"].corr(aligned["actual"], method="spearman"))
            ic_by_lag[lag] = round(ic, 6)

        return ic_by_lag

    # ------------------------------------------------------------------
    # Return autocorrelation
    # ------------------------------------------------------------------

    def compute_return_autocorrelation(
        self,
        predictions: pd.Series,
        actuals: pd.Series,
    ) -> dict[int, float]:
        """Compute autocorrelation of signal returns at each lag.

        Signal return at time *t* is ``sign(pred[t]) * actual[t]``.

        Parameters
        ----------
        predictions:
            Alpha scores.
        actuals:
            Realised returns.

        Returns
        -------
        dict[int, float]
            ``{lag: autocorrelation}``
        """
        signal_returns = np.sign(predictions) * actuals
        autocorr: dict[int, float] = {}

        for lag in range(1, self.max_lag + 1):
            if len(signal_returns) <= lag:
                break
            ac = float(signal_returns.autocorr(lag=lag))
            if np.isfinite(ac):
                autocorr[lag] = round(ac, 6)

        return autocorr

    # ------------------------------------------------------------------
    # Half-life estimation
    # ------------------------------------------------------------------

    def estimate_halflife(self, ic_by_lag: dict[int, float]) -> float:
        """Estimate the half-life of alpha decay from the IC curve.

        The half-life is the lag at which the IC drops to 50% of its
        peak value.  If the IC never drops below 50%, returns ``inf``.

        Parameters
        ----------
        ic_by_lag:
            IC values by lag.

        Returns
        -------
        float
            Estimated half-life in bars.
        """
        if not ic_by_lag:
            return float("inf")

        lags = sorted(ic_by_lag.keys())
        ics = [ic_by_lag[lag] for lag in lags]

        peak_ic = max(ics)
        if peak_ic <= 0:
            return 0.0

        threshold = peak_ic * 0.5

        for lag, ic in zip(lags, ics):
            if ic < threshold:
                return float(lag)

        return float("inf")

    # ------------------------------------------------------------------
    # Decay rate
    # ------------------------------------------------------------------

    def _decay_rate(self, ic_by_lag: dict[int, float]) -> float:
        """Estimate exponential decay rate from IC curve.

        Fits ``IC(lag) = IC(0) * exp(-rate * lag)`` via log-linear regression.

        Parameters
        ----------
        ic_by_lag:
            IC values by lag.

        Returns
        -------
        float
            Decay rate (higher = faster decay).
        """
        if len(ic_by_lag) < 3:
            return 0.0

        lags = np.array(sorted(ic_by_lag.keys()), dtype=float)
        ics = np.array([ic_by_lag[int(lag)] for lag in lags])

        # Only use positive ICs for log fit
        mask = ics > 1e-6
        if mask.sum() < 3:
            return 0.0

        log_ics = np.log(ics[mask])
        lag_masked = lags[mask]

        # Linear regression: log(IC) = a - rate * lag
        if len(lag_masked) < 2:
            return 0.0

        coeffs = np.polyfit(lag_masked, log_ics, deg=1)
        rate = float(-coeffs[0])
        return round(max(rate, 0.0), 6)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def decay_report_df(
        self,
        predictions: pd.Series,
        actuals: pd.Series,
    ) -> pd.DataFrame:
        """Generate a decay report as a DataFrame.

        Parameters
        ----------
        predictions:
            Alpha scores.
        actuals:
            Realised returns.

        Returns
        -------
        pd.DataFrame
            One row per lag with columns: ``lag``, ``ic``, ``autocorrelation``.
        """
        ic_by_lag = self.compute_ic_decay(predictions, actuals)
        autocorr = self.compute_return_autocorrelation(predictions, actuals)

        rows = []
        for lag in sorted(set(list(ic_by_lag.keys()) + list(autocorr.keys()))):
            rows.append({
                "lag": lag,
                "ic": ic_by_lag.get(lag, float("nan")),
                "autocorrelation": autocorr.get(lag, float("nan")),
            })

        return pd.DataFrame(rows)
