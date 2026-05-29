"""
aqc/indicators/volatility.py
============================
Vectorised volatility indicators: Bollinger Bands and ATR.

Author: AQC Team
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from aqc.indicators.moving_averages import sma


def bollinger_bands(
    series: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands.

    Parameters
    ----------
    series:
        Price series (typically ``close``).
    period:
        Rolling window for the middle band (SMA).
    std_dev:
        Number of standard deviations for the upper and lower bands.

    Returns
    -------
    tuple[pd.Series, pd.Series, pd.Series]
        ``(upper_band, middle_band, lower_band)``

    Examples
    --------
    >>> upper, mid, lower = bollinger_bands(df["close"], 20, 2.0)
    """
    middle = sma(series, period)
    rolling_std = series.rolling(window=period, min_periods=period).std()
    upper = middle + std_dev * rolling_std
    lower = middle - std_dev * rolling_std
    return upper, middle, lower


def bollinger_bandwidth(
    series: pd.Series, period: int = 20, std_dev: float = 2.0
) -> pd.Series:
    """Bollinger Band Width — normalised measure of band width.

    ``Bandwidth = (Upper - Lower) / Middle``

    Parameters
    ----------
    series:
        Price series.
    period:
        Bollinger Band period.
    std_dev:
        Standard deviation multiplier.

    Returns
    -------
    pd.Series
    """
    upper, middle, lower = bollinger_bands(series, period, std_dev)
    return (upper - lower) / middle


def bollinger_percent_b(
    series: pd.Series, period: int = 20, std_dev: float = 2.0
) -> pd.Series:
    r"""%B — position of price within the Bollinger Bands.

    ``%B = (Price - Lower) / (Upper - Lower)``

    Values:
    * ``%B > 1``: price above upper band.
    * ``%B = 0.5``: price at the middle band.
    * ``%B < 0``: price below lower band.

    Parameters
    ----------
    series:
        Price series.
    period:
        Bollinger Band period.
    std_dev:
        Standard deviation multiplier.

    Returns
    -------
    pd.Series
    """
    upper, _, lower = bollinger_bands(series, period, std_dev)
    band_width = (upper - lower).replace(0, float("nan"))
    return (series - lower) / band_width


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Average True Range (ATR).

    The True Range for each bar is:

    ``TR = max(High - Low, |High - Prev Close|, |Low - Prev Close|)``

    ATR is the Wilder smoothed average of TR.

    Parameters
    ----------
    high:
        High price series.
    low:
        Low price series.
    close:
        Close price series.
    period:
        ATR period (default 14).

    Returns
    -------
    pd.Series

    Examples
    --------
    >>> atr_series = atr(df["high"], df["low"], df["close"], 14)
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def historical_volatility(
    series: pd.Series,
    period: int = 21,
    annualisation_factor: int = 252,
) -> pd.Series:
    """Realised (historical) volatility.

    Computed as the rolling standard deviation of log-returns, annualised.

    Parameters
    ----------
    series:
        Price series.
    period:
        Rolling window (default 21 trading days ≈ 1 month).
    annualisation_factor:
        252 for daily, 52 for weekly, 12 for monthly.

    Returns
    -------
    pd.Series
        Annualised volatility as a decimal (e.g. ``0.20`` = 20%).
    """
    log_ret = np.log(series / series.shift(1))
    return log_ret.rolling(period).std() * np.sqrt(annualisation_factor)
