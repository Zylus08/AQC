"""
aqc/indicators/moving_averages.py
=================================
Vectorised moving average indicators.

All functions operate on :class:`~pandas.Series` objects and return a
:class:`~pandas.Series` of the same length.  All use pandas' built-in
rolling operations for correctness and performance.

Author: AQC Team
"""

from __future__ import annotations

import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average.

    Parameters
    ----------
    series:
        Price series (typically ``close``).
    period:
        Rolling window size in bars.

    Returns
    -------
    pd.Series
        SMA values (first ``period - 1`` values are ``NaN``).

    Examples
    --------
    >>> sma(df["close"], 20)
    """
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int, adjust: bool = False) -> pd.Series:
    """Exponential Moving Average.

    Uses a smoothing factor of ``2 / (period + 1)``.

    Parameters
    ----------
    series:
        Price series.
    period:
        EMA span.
    adjust:
        If ``True``, use pandas' adjustment to correct for early observations.
        Defaults to ``False`` (matches most charting platforms).

    Returns
    -------
    pd.Series

    Examples
    --------
    >>> ema(df["close"], 12)
    """
    return series.ewm(span=period, adjust=adjust, min_periods=period).mean()


def wma(series: pd.Series, period: int) -> pd.Series:
    """Linearly Weighted Moving Average.

    Weights decrease linearly: the most recent bar has weight *period*,
    the previous bar has weight *period - 1*, …, the oldest has weight 1.

    Parameters
    ----------
    series:
        Price series.
    period:
        Rolling window size.

    Returns
    -------
    pd.Series

    Examples
    --------
    >>> wma(df["close"], 14)
    """
    weights = pd.Series(range(1, period + 1), dtype=float)
    total_weight = weights.sum()

    def _weighted_mean(window: pd.Series) -> float:
        return (window.values * weights.values[-len(window):]).sum() / total_weight

    return series.rolling(window=period, min_periods=period).apply(_weighted_mean, raw=False)


def dema(series: pd.Series, period: int) -> pd.Series:
    """Double Exponential Moving Average (DEMA).

    ``DEMA = 2 * EMA(n) - EMA(EMA(n))``

    Reduces the lag of a standard EMA.

    Parameters
    ----------
    series:
        Price series.
    period:
        EMA period.

    Returns
    -------
    pd.Series
    """
    e1 = ema(series, period)
    e2 = ema(e1, period)
    return 2 * e1 - e2


def hull_ma(series: pd.Series, period: int) -> pd.Series:
    """Hull Moving Average (HMA).

    ``HMA(n) = WMA(2 * WMA(n/2) − WMA(n), sqrt(n))``

    Significantly reduces lag while maintaining smoothness.

    Parameters
    ----------
    series:
        Price series.
    period:
        HMA period.

    Returns
    -------
    pd.Series
    """
    import math
    half = max(1, period // 2)
    sqrt_n = max(1, int(math.sqrt(period)))
    raw = 2 * wma(series, half) - wma(series, period)
    return wma(raw, sqrt_n)
