"""
aqc/regimes/volatility_regime.py
=================================
Volatility Regime Detector.

Classifies the current volatility environment into one of four states:

    LOW → NORMAL → HIGH → EXTREME

using a rolling-percentile methodology on realised or forecast volatility.

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


class VolatilityRegime(Enum):
    """Volatility regime classification."""

    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


@dataclass
class VolRegimeThresholds:
    """Percentile thresholds for regime boundaries.

    Attributes
    ----------
    low:
        Percentile below which vol is LOW (default 25th).
    high:
        Percentile above which vol is HIGH (default 75th).
    extreme:
        Percentile above which vol is EXTREME (default 95th).
    """

    low: float = 25.0
    high: float = 75.0
    extreme: float = 95.0


class VolatilityRegimeDetector:
    """Detects volatility regimes using rolling percentile methodology.

    Parameters
    ----------
    window:
        Rolling window for historical vol computation (default 21).
    history_length:
        Number of vol observations to retain for percentile computation
        (default 252 ≈ 1 year).
    ann_factor:
        Annualisation factor (default 252 for daily).
    thresholds:
        Percentile boundaries.

    Examples
    --------
    >>> detector = VolatilityRegimeDetector()
    >>> regime = detector.detect(close_prices)
    """

    def __init__(
        self,
        window: int = 21,
        history_length: int = 252,
        ann_factor: int = 252,
        thresholds: Optional[VolRegimeThresholds] = None,
    ) -> None:
        self.window = window
        self.history_length = history_length
        self.ann_factor = ann_factor
        self.thresholds = thresholds or VolRegimeThresholds()
        self._vol_history: deque[float] = deque(maxlen=history_length)

    def detect(self, prices: pd.Series) -> VolatilityRegime:
        """Classify current volatility regime from a price series.

        Parameters
        ----------
        prices:
            Close price series.

        Returns
        -------
        VolatilityRegime
        """
        vol = self._compute_current_vol(prices)
        if np.isnan(vol):
            return VolatilityRegime.NORMAL

        self._vol_history.append(vol)
        return self._classify(vol)

    def detect_series(self, prices: pd.Series) -> pd.DataFrame:
        """Compute volatility regime for every bar in a price series.

        Parameters
        ----------
        prices:
            Close price series.

        Returns
        -------
        pd.DataFrame
            Columns: ``vol``, ``regime``.
        """
        log_ret = np.log(prices / prices.shift(1)).dropna()
        rolling_vol = log_ret.rolling(self.window).std() * np.sqrt(self.ann_factor)

        regimes = []
        self._vol_history.clear()

        for ts, vol in rolling_vol.items():
            if np.isnan(vol):
                regimes.append(VolatilityRegime.NORMAL.value)
                continue
            self._vol_history.append(vol)
            regimes.append(self._classify(vol).value)

        return pd.DataFrame(
            {"vol": rolling_vol.values, "regime": regimes},
            index=rolling_vol.index,
        )

    def _compute_current_vol(self, prices: pd.Series) -> float:
        """Compute the latest realised volatility."""
        if len(prices) < self.window + 1:
            return float("nan")
        log_ret = np.log(prices / prices.shift(1)).dropna()
        return float(log_ret.iloc[-self.window:].std() * np.sqrt(self.ann_factor))

    def _classify(self, vol: float) -> VolatilityRegime:
        """Classify a vol reading against the running distribution."""
        if len(self._vol_history) < 10:
            return VolatilityRegime.NORMAL

        arr = np.array(self._vol_history)
        p_low = np.percentile(arr, self.thresholds.low)
        p_high = np.percentile(arr, self.thresholds.high)
        p_extreme = np.percentile(arr, self.thresholds.extreme)

        if vol >= p_extreme:
            return VolatilityRegime.EXTREME
        elif vol >= p_high:
            return VolatilityRegime.HIGH
        elif vol <= p_low:
            return VolatilityRegime.LOW
        else:
            return VolatilityRegime.NORMAL
