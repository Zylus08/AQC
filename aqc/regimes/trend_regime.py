"""
aqc/regimes/trend_regime.py
============================
Trend Regime Detector.

Classifies the current market trend into five states:

    STRONG_DOWNTREND → DOWNTREND → RANGE_BOUND → UPTREND → STRONG_UPTREND

Methods:
- Moving average slope (primary)
- ADX strength filter
- Price position relative to moving averages

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TrendRegime(Enum):
    """Trend regime classification."""

    STRONG_UPTREND = "STRONG_UPTREND"
    UPTREND = "UPTREND"
    RANGE_BOUND = "RANGE_BOUND"
    DOWNTREND = "DOWNTREND"
    STRONG_DOWNTREND = "STRONG_DOWNTREND"


@dataclass
class TrendConfig:
    """Configuration for trend detection.

    Attributes
    ----------
    short_ma:
        Short moving average period (default 20).
    long_ma:
        Long moving average period (default 50).
    slope_window:
        Window for computing MA slope (default 10).
    adx_period:
        ADX look-back period (default 14).
    adx_strong_threshold:
        ADX above this → strong trend (default 30).
    adx_weak_threshold:
        ADX below this → range-bound (default 20).
    slope_strong_threshold:
        Normalised slope above this → strong trend.
    slope_weak_threshold:
        Normalised slope below this → range-bound.
    """

    short_ma: int = 20
    long_ma: int = 50
    slope_window: int = 10
    adx_period: int = 14
    adx_strong_threshold: float = 30.0
    adx_weak_threshold: float = 20.0
    slope_strong_threshold: float = 0.002
    slope_weak_threshold: float = 0.0005


class TrendRegimeDetector:
    """Detects trend regimes using MA slope and ADX.

    Parameters
    ----------
    config:
        Trend detection configuration.

    Examples
    --------
    >>> detector = TrendRegimeDetector()
    >>> regime = detector.detect(df)  # df has OHLC columns
    """

    def __init__(self, config: Optional[TrendConfig] = None) -> None:
        self.config = config or TrendConfig()

    def detect(self, df: pd.DataFrame) -> TrendRegime:
        """Classify current trend from OHLC data.

        Parameters
        ----------
        df:
            DataFrame with ``close``, ``high``, ``low`` columns.

        Returns
        -------
        TrendRegime
        """
        if len(df) < self.config.long_ma + self.config.slope_window:
            return TrendRegime.RANGE_BOUND

        close = df["close"]

        # MA slope
        ma = close.rolling(self.config.short_ma).mean()
        slope = self._normalised_slope(ma, self.config.slope_window)

        # ADX
        adx_val = self._compute_adx(df)

        return self._classify(slope, adx_val)

    def detect_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute trend regime for every bar.

        Parameters
        ----------
        df:
            DataFrame with ``close``, ``high``, ``low`` columns.

        Returns
        -------
        pd.DataFrame
            Columns: ``ma_slope``, ``adx``, ``regime``.
        """
        close = df["close"]
        cfg = self.config

        # Rolling MA
        ma = close.rolling(cfg.short_ma).mean()

        # Rolling normalised slope
        slopes = pd.Series(np.nan, index=df.index)
        for i in range(cfg.short_ma + cfg.slope_window - 1, len(df)):
            segment = ma.iloc[i - cfg.slope_window + 1 : i + 1].dropna()
            if len(segment) >= 2:
                s = (segment.iloc[-1] - segment.iloc[0]) / (segment.iloc[0] * len(segment))
                slopes.iloc[i] = s

        # Rolling ADX
        adx_series = self._compute_adx_series(df)

        # Classify
        regimes = []
        for i in range(len(df)):
            s = slopes.iloc[i] if not np.isnan(slopes.iloc[i]) else 0.0
            a = adx_series.iloc[i] if i < len(adx_series) and not np.isnan(adx_series.iloc[i]) else 0.0
            regimes.append(self._classify(s, a).value)

        return pd.DataFrame(
            {"ma_slope": slopes.values, "adx": adx_series.values, "regime": regimes},
            index=df.index,
        )

    def _normalised_slope(self, ma: pd.Series, window: int) -> float:
        """Compute normalised slope of the last `window` MA values."""
        segment = ma.dropna().iloc[-window:]
        if len(segment) < 2:
            return 0.0
        return float((segment.iloc[-1] - segment.iloc[0]) / (segment.iloc[0] * len(segment)))

    def _compute_adx(self, df: pd.DataFrame) -> float:
        """Compute the latest ADX value."""
        adx_series = self._compute_adx_series(df)
        valid = adx_series.dropna()
        return float(valid.iloc[-1]) if len(valid) > 0 else 0.0

    def _compute_adx_series(self, df: pd.DataFrame) -> pd.Series:
        """Compute the full ADX series.

        ADX = smoothed average of DX.
        DX = |+DI - -DI| / (+DI + -DI)
        """
        period = self.config.adx_period
        high = df["high"]
        low = df["low"]
        close = df["close"]

        plus_dm = high.diff().clip(lower=0)
        minus_dm = (-low.diff()).clip(lower=0)

        # Zero out when opposite DM is larger
        both = pd.DataFrame({"p": plus_dm, "m": minus_dm})
        plus_dm = both.apply(lambda r: r["p"] if r["p"] > r["m"] else 0, axis=1)
        minus_dm = both.apply(lambda r: r["m"] if r["m"] > r["p"] else 0, axis=1)

        # True Range
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)

        # Smoothed with Wilder's method
        atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        smooth_plus = plus_dm.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        smooth_minus = minus_dm.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

        plus_di = 100.0 * smooth_plus / atr.replace(0, np.nan)
        minus_di = 100.0 * smooth_minus / atr.replace(0, np.nan)

        dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

        return adx

    def _classify(self, slope: float, adx: float) -> TrendRegime:
        """Classify trend from slope and ADX."""
        cfg = self.config

        if adx < cfg.adx_weak_threshold and abs(slope) < cfg.slope_weak_threshold:
            return TrendRegime.RANGE_BOUND

        if slope > cfg.slope_strong_threshold and adx > cfg.adx_strong_threshold:
            return TrendRegime.STRONG_UPTREND
        elif slope > cfg.slope_weak_threshold:
            return TrendRegime.UPTREND
        elif slope < -cfg.slope_strong_threshold and adx > cfg.adx_strong_threshold:
            return TrendRegime.STRONG_DOWNTREND
        elif slope < -cfg.slope_weak_threshold:
            return TrendRegime.DOWNTREND
        else:
            return TrendRegime.RANGE_BOUND
