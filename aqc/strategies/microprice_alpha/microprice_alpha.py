"""
aqc/strategies/microprice_alpha/microprice_alpha.py
=====================================================
Microprice Alpha implementation.

Predicts near-term mean reversion based on microprice deviations.
When microprice is significantly higher than mid-price, it suggests
upward pressure (buy). When lower, downward pressure (sell).

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from aqc.alpha import (
    AlphaBase,
    AlphaCategory,
    AlphaFrequency,
    AlphaMetadata,
    AlphaMetrics,
    AlphaSignal,
    register_alpha,
)
from aqc.strategies.microprice_alpha.microprice_features import MicropriceFeatureEngine

logger = logging.getLogger(__name__)


@register_alpha(name="MicropriceAlpha", category=AlphaCategory.MICROPRICE)
class MicropriceAlpha(AlphaBase):
    """Generates signals based on microprice deviations.

    Parameters
    ----------
    name:
        Alpha identifier.
    zscore_threshold:
        Threshold for deviation z-score to trigger a signal.
    n_levels:
        Order book depth levels to consider for fair value.
    zscore_window:
        Rolling window for z-score computation.
    """

    def __init__(
        self,
        name: str = "microprice_alpha",
        version: str = "1.0.0",
        zscore_threshold: float = 1.5,
        n_levels: int = 5,
        zscore_window: int = 20,
    ) -> None:
        super().__init__(name=name, version=version)
        self.zscore_threshold = zscore_threshold
        self.n_levels = n_levels
        self.zscore_window = zscore_window

        self._feature_engine = MicropriceFeatureEngine(
            n_levels=n_levels, zscore_window=zscore_window
        )
        self._is_fitted = True  # This is a heuristic alpha, no ML training required

    def metadata(self) -> AlphaMetadata:
        return AlphaMetadata(
            name=self.name,
            version=self.version,
            author="AQC Team",
            category=AlphaCategory.MICROPRICE,
            frequency=AlphaFrequency.TICK,
            description="Predicts near-term returns based on microprice deviation from mid-price.",
            parameters={
                "zscore_threshold": self.zscore_threshold,
                "n_levels": self.n_levels,
                "zscore_window": self.zscore_window,
            },
        )

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Produce continuous score based on deviation z-score."""
        zscore = features["deviation_zscore"].fillna(0.0)
        # Scale score so that threshold corresponds to roughly 0.5 score
        score = zscore / (self.zscore_threshold * 2)
        return score.clip(-1.0, 1.0)

    def generate_signal(self, data: pd.DataFrame) -> AlphaSignal:
        """Generate a signal from raw snapshot data."""
        # Need at least zscore_window bars to compute a valid z-score
        if len(data) < self.zscore_window:
            return AlphaSignal(alpha_name=self.name, direction=0)

        latest = data.iloc[[-1]]
        features = self._feature_engine.extract(data.tail(self.zscore_window))
        
        zscore = float(features["deviation_zscore"].iloc[-1])
        score = max(-1.0, min(1.0, zscore / (self.zscore_threshold * 2)))

        if zscore > self.zscore_threshold:
            direction = 1
        elif zscore < -self.zscore_threshold:
            direction = -1
        else:
            direction = 0

        confidence = min(abs(zscore) / (self.zscore_threshold * 2), 1.0)

        return AlphaSignal(
            alpha_name=self.name,
            timestamp=latest.index[0],
            direction=direction,
            strength=score,
            confidence=confidence,
            features_used=("deviation_zscore",),
        )

    def evaluate(self, predictions: pd.Series, actuals: pd.Series) -> AlphaMetrics:
        """Evaluate out-of-sample performance."""
        # Convert continuous score to directional signals based on threshold equivalent
        score_threshold = 0.5  # since score = zscore / (threshold * 2)
        signals = predictions.copy()
        signals[predictions > score_threshold] = 1
        signals[predictions < -score_threshold] = -1
        signals[predictions.abs() <= score_threshold] = 0

        metrics = AlphaMetrics.from_predictions(signals, actuals)
        self.cache_metrics(metrics)
        return metrics
