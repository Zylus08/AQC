"""
aqc/regimes/correlation_regime.py
==================================
Correlation Regime Detector.

Classifies the cross-asset correlation environment:

    LOW_CORRELATION → NORMAL_CORRELATION → HIGH_CORRELATION → CRISIS_CORRELATION

Crisis correlation is characterised by all assets moving together
(typically down) — a hallmark of market stress events.

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


class CorrelationRegime(Enum):
    """Correlation regime classification."""

    LOW_CORRELATION = "LOW_CORRELATION"
    NORMAL_CORRELATION = "NORMAL_CORRELATION"
    HIGH_CORRELATION = "HIGH_CORRELATION"
    CRISIS_CORRELATION = "CRISIS_CORRELATION"


@dataclass
class CorrelationConfig:
    """Configuration for correlation regime detection.

    Attributes
    ----------
    window:
        Rolling window for pairwise correlation (default 63 ≈ 3 months).
    history_length:
        Observations retained for percentile thresholds (default 252).
    low_pct:
        Percentile below which → LOW (default 25).
    high_pct:
        Percentile above which → HIGH (default 75).
    crisis_pct:
        Percentile above which → CRISIS (default 95).
    """

    window: int = 63
    history_length: int = 252
    low_pct: float = 25.0
    high_pct: float = 75.0
    crisis_pct: float = 95.0


class CorrelationRegimeDetector:
    """Detects correlation regimes from multi-asset returns.

    Parameters
    ----------
    config:
        Correlation detection configuration.

    Examples
    --------
    >>> detector = CorrelationRegimeDetector()
    >>> regime = detector.detect(returns_df)  # columns = symbols
    """

    def __init__(self, config: Optional[CorrelationConfig] = None) -> None:
        self.config = config or CorrelationConfig()
        self._avg_corr_history: deque[float] = deque(maxlen=self.config.history_length)

    def detect(self, returns: pd.DataFrame) -> CorrelationRegime:
        """Classify current correlation regime.

        Parameters
        ----------
        returns:
            DataFrame of returns with columns as symbols.

        Returns
        -------
        CorrelationRegime
        """
        avg_corr = self._compute_avg_correlation(returns)
        if np.isnan(avg_corr):
            return CorrelationRegime.NORMAL_CORRELATION

        self._avg_corr_history.append(avg_corr)
        return self._classify(avg_corr)

    def detect_series(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Compute correlation regime for every bar.

        Parameters
        ----------
        returns:
            Multi-column returns DataFrame.

        Returns
        -------
        pd.DataFrame
            Columns: ``avg_corr``, ``regime``.
        """
        if returns.shape[1] < 2:
            return pd.DataFrame(
                {"avg_corr": np.nan, "regime": CorrelationRegime.NORMAL_CORRELATION.value},
                index=returns.index,
            )

        window = self.config.window
        n = len(returns)
        avg_corrs = pd.Series(np.nan, index=returns.index)
        regimes = []

        self._avg_corr_history.clear()

        for i in range(window, n):
            chunk = returns.iloc[i - window : i]
            corr_matrix = chunk.corr()

            # Average of upper-triangle off-diagonal elements
            mask = np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
            upper = corr_matrix.values[mask]
            valid = upper[~np.isnan(upper)]

            if len(valid) > 0:
                avg_c = float(np.mean(valid))
            else:
                avg_c = float("nan")

            avg_corrs.iloc[i] = avg_c

            if not np.isnan(avg_c):
                self._avg_corr_history.append(avg_c)

        # Now classify
        self._avg_corr_history.clear()
        for i in range(n):
            c = avg_corrs.iloc[i]
            if np.isnan(c):
                regimes.append(CorrelationRegime.NORMAL_CORRELATION.value)
            else:
                self._avg_corr_history.append(c)
                regimes.append(self._classify(c).value)

        return pd.DataFrame(
            {"avg_corr": avg_corrs.values, "regime": regimes},
            index=returns.index,
        )

    def _compute_avg_correlation(self, returns: pd.DataFrame) -> float:
        """Compute average pairwise correlation from trailing window."""
        if returns.shape[1] < 2 or len(returns) < self.config.window:
            return float("nan")

        chunk = returns.iloc[-self.config.window:]
        corr_matrix = chunk.corr()
        mask = np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
        upper = corr_matrix.values[mask]
        valid = upper[~np.isnan(upper)]

        return float(np.mean(valid)) if len(valid) > 0 else float("nan")

    def _classify(self, avg_corr: float) -> CorrelationRegime:
        """Classify using running percentiles."""
        if len(self._avg_corr_history) < 10:
            return CorrelationRegime.NORMAL_CORRELATION

        arr = np.array(self._avg_corr_history)
        p_low = np.percentile(arr, self.config.low_pct)
        p_high = np.percentile(arr, self.config.high_pct)
        p_crisis = np.percentile(arr, self.config.crisis_pct)

        if avg_corr >= p_crisis:
            return CorrelationRegime.CRISIS_CORRELATION
        elif avg_corr >= p_high:
            return CorrelationRegime.HIGH_CORRELATION
        elif avg_corr <= p_low:
            return CorrelationRegime.LOW_CORRELATION
        else:
            return CorrelationRegime.NORMAL_CORRELATION
