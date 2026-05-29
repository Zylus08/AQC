"""
aqc/indicators/momentum.py
==========================
Vectorised momentum indicators: RSI and MACD.

Author: AQC Team
"""

from __future__ import annotations

import pandas as pd

from aqc.indicators.moving_averages import ema


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (RSI).

    Uses Wilder's smoothing (equivalent to EMA with ``alpha = 1 / period``).

    Parameters
    ----------
    series:
        Price series (typically ``close``).
    period:
        Look-back period (default 14).

    Returns
    -------
    pd.Series
        RSI values in the range ``[0, 100]``.

    Examples
    --------
    >>> rsi(df["close"], 14)
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder smoothing = EMA with alpha = 1/period (i.e. span = 2*period - 1)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, float("nan"))
    return 100.0 - (100.0 / (1.0 + rs))


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Moving Average Convergence Divergence (MACD).

    Parameters
    ----------
    series:
        Price series (typically ``close``).
    fast:
        Fast EMA period (default 12).
    slow:
        Slow EMA period (default 26).
    signal:
        Signal line EMA period (default 9).

    Returns
    -------
    tuple[pd.Series, pd.Series, pd.Series]
        ``(macd_line, signal_line, histogram)``

    Examples
    --------
    >>> macd_line, sig_line, hist = macd(df["close"])
    """
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """Stochastic Oscillator.

    Parameters
    ----------
    high:
        High price series.
    low:
        Low price series.
    close:
        Close price series.
    k_period:
        %K look-back period.
    d_period:
        %D smoothing period.

    Returns
    -------
    tuple[pd.Series, pd.Series]
        ``(%K, %D)``
    """
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    k = 100.0 * (close - lowest_low) / (highest_high - lowest_low).replace(0, float("nan"))
    d = k.rolling(d_period).mean()
    return k, d


def rate_of_change(series: pd.Series, period: int = 14) -> pd.Series:
    """Rate of Change (ROC) / momentum.

    ``ROC(n) = (close / close[n]) - 1``

    Parameters
    ----------
    series:
        Price series.
    period:
        Look-back period.

    Returns
    -------
    pd.Series
        ROC as a decimal (not percentage).
    """
    return series.pct_change(periods=period)
