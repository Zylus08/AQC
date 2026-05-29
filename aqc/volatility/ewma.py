"""
aqc/volatility/ewma.py
=======================
Exponentially Weighted Moving Average (EWMA) Volatility Estimator.

Implements the RiskMetrics EWMA model for variance estimation:

    sigma^2_t = lambda * sigma^2_{t-1} + (1 - lambda) * r^2_{t-1}

The EWMA model has one parameter — the decay factor ``lambda`` (typically
0.94 for daily data, 0.97 for monthly).  Unlike GARCH, there is no
mean-reversion term, making EWMA a special case of IGARCH(1,1).

Advantages over rolling window:
- Reacts faster to recent volatility changes
- Exponential weighting captures volatility clustering
- Single parameter, no window-length sensitivity

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def ewma_variance(
    returns: pd.Series,
    decay: float = 0.94,
    min_periods: int = 10,
) -> pd.Series:
    """Compute EWMA variance of a return series.

    Uses the RiskMetrics recursion:

        var_t = decay * var_{t-1} + (1 - decay) * r^2_{t-1}

    Parameters
    ----------
    returns:
        Return series (typically log-returns or simple returns).
    decay:
        Exponential decay factor (lambda). Standard values:
        * ``0.94`` for daily data (RiskMetrics).
        * ``0.97`` for monthly data.
    min_periods:
        Minimum number of observations before producing a value.

    Returns
    -------
    pd.Series
        EWMA variance series.

    Examples
    --------
    >>> log_ret = np.log(df["close"] / df["close"].shift(1))
    >>> var = ewma_variance(log_ret, decay=0.94)
    """
    if not 0 < decay < 1:
        raise ValueError(f"decay must be in (0, 1), got {decay}")

    # pandas EWM with com = decay / (1 - decay) gives equivalent weighting
    # But we use explicit recursion for clarity and correctness
    n = len(returns)
    variance = np.full(n, np.nan)

    # Seed with sample variance of first min_periods observations
    valid = returns.dropna()
    if len(valid) < min_periods:
        return pd.Series(variance, index=returns.index, name="ewma_variance")

    first_valid_idx = valid.index[0]
    seed_start = returns.index.get_loc(first_valid_idx)
    seed_end = seed_start + min_periods

    if seed_end > n:
        return pd.Series(variance, index=returns.index, name="ewma_variance")

    seed_var = float(returns.iloc[seed_start:seed_end].var())
    variance[seed_end - 1] = seed_var

    for i in range(seed_end, n):
        r_prev = returns.iloc[i - 1]
        if np.isnan(r_prev) or np.isnan(variance[i - 1]):
            variance[i] = variance[i - 1] if not np.isnan(variance[i - 1]) else np.nan
        else:
            variance[i] = decay * variance[i - 1] + (1 - decay) * r_prev ** 2

    return pd.Series(variance, index=returns.index, name="ewma_variance")


def ewma_volatility(
    returns: pd.Series,
    decay: float = 0.94,
    min_periods: int = 10,
    annualise: bool = True,
    ann_factor: int = 252,
) -> pd.Series:
    """Compute EWMA volatility (annualised standard deviation).

    Parameters
    ----------
    returns:
        Return series.
    decay:
        Exponential decay factor (default 0.94).
    min_periods:
        Warm-up observations.
    annualise:
        If True, multiply by ``sqrt(ann_factor)``.
    ann_factor:
        Annualisation factor (252 for daily, 52 for weekly).

    Returns
    -------
    pd.Series
        EWMA volatility series.

    Examples
    --------
    >>> vol = ewma_volatility(log_returns, decay=0.94)
    """
    var = ewma_variance(returns, decay=decay, min_periods=min_periods)
    vol = np.sqrt(var)
    if annualise:
        vol = vol * np.sqrt(ann_factor)
    result = pd.Series(vol, index=returns.index, name="ewma_volatility")
    return result


def ewma_forecast(
    returns: pd.Series,
    decay: float = 0.94,
    horizon: int = 1,
    min_periods: int = 10,
    ann_factor: int = 252,
) -> pd.Series:
    """Forecast EWMA volatility for the next ``horizon`` periods.

    Under EWMA (IGARCH), the h-step-ahead variance forecast is:

        var_{t+h|t} = h * var_{t+1|t}

    So the h-step volatility is ``sqrt(h) * sigma_{t+1|t}``.

    Parameters
    ----------
    returns:
        Return series.
    decay:
        Decay factor.
    horizon:
        Forecast horizon in periods (default 1).
    min_periods:
        Warm-up.
    ann_factor:
        Annualisation factor.

    Returns
    -------
    pd.Series
        Annualised h-step-ahead volatility forecast.
    """
    var = ewma_variance(returns, decay=decay, min_periods=min_periods)
    forecast_var = var * horizon
    return pd.Series(
        np.sqrt(forecast_var) * np.sqrt(ann_factor / horizon),
        index=returns.index,
        name=f"ewma_forecast_{horizon}d",
    )
