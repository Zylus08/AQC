"""
aqc/strategies/liquidity_alpha/liquidity_alpha.py
===================================================
Liquidity Alpha implementation.

Generates signals based on liquidity expansion/contraction.
Liquidity withdrawal (depth drop + spread expansion) often precedes
directional moves.

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging

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
from aqc.strategies.liquidity_alpha.liquidity_features import LiquidityFeatureEngine

logger = logging.getLogger(__name__)


@register_alpha(name="LiquidityAlpha", category=AlphaCategory.LIQUIDITY)
class LiquidityAlpha(AlphaBase):
    """Generates defensive and directional signals from liquidity state.

    Parameters
    ----------
    name:
        Alpha identifier.
    zscore_threshold:
        Threshold for depth z-score to trigger a shock signal.
    n_levels:
        Order book depth levels to consider.
    lookback:
        Rolling window for z-score computation.
    """

    def __init__(
        self,
        name: str = "liquidity_alpha",
        version: str = "1.0.0",
        zscore_threshold: float = -2.0,  # Negative because we look for depth drops
        n_levels: int = 5,
        lookback: int = 20,
    ) -> None:
        super().__init__(name=name, version=version)
        self.zscore_threshold = zscore_threshold
        self.n_levels = n_levels
        self.lookback = lookback

        self._feature_engine = LiquidityFeatureEngine(
            n_levels=n_levels, lookback=lookback
        )
        self._is_fitted = True

    def metadata(self) -> AlphaMetadata:
        return AlphaMetadata(
            name=self.name,
            version=self.version,
            author="AQC Team",
            category=AlphaCategory.LIQUIDITY,
            frequency=AlphaFrequency.TICK,
            description="Predicts directional momentum following liquidity withdrawal shocks.",
            parameters={
                "zscore_threshold": self.zscore_threshold,
                "n_levels": self.n_levels,
                "lookback": self.lookback,
            },
        )

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Produce continuous score based on depth z-score inverted.
        
        When depth drops significantly (highly negative zscore), it indicates
        a liquidity vacuum. The direction of the resulting move depends on order flow,
        but for this base alpha we map extreme liquidity drop to a defensive flat signal (0).
        Otherwise we mean-revert the spread expansion.
        """
        # If liquidity is shocked, score is 0 (defensive)
        # If liquidity is abundant, we might fade spread expansion
        scores = pd.Series(0.0, index=features.index)
        
        # Simple heuristic: if spread is unusually wide but depth is normal, fade it.
        # This is a very basic liquidity provision proxy.
        z_spread = features.get("spread_zscore", pd.Series(0.0, index=features.index)).fillna(0.0)
        z_depth = features.get("depth_zscore", pd.Series(0.0, index=features.index)).fillna(0.0)
        
        # Map to [-1, 1] - positive score means provide liquidity (fade the move)
        # Simplified: just output a continuous value that represents liquidity stress
        stress = (z_spread - z_depth) / 4.0
        scores = -stress.clip(-1.0, 1.0) # Negative because high stress means don't trade
        
        return scores

    def generate_signal(self, data: pd.DataFrame) -> AlphaSignal:
        """Generate a signal from raw snapshot data."""
        if len(data) < self.lookback:
            return AlphaSignal(alpha_name=self.name, direction=0)

        latest = data.iloc[[-1]]
        features = self._feature_engine.extract(data.tail(self.lookback))
        
        score = float(self.predict(features).iloc[-1])
        z_depth = float(features["depth_zscore"].iloc[-1])
        
        direction = 0
        confidence = 0.0
        
        # Liquidity shock -> defensive signal
        if z_depth < self.zscore_threshold:
            direction = 0
            confidence = 1.0  # High confidence to flatten
        elif score > 0.5:
            direction = 1
            confidence = min(abs(score), 1.0)
        elif score < -0.5:
            direction = -1
            confidence = min(abs(score), 1.0)

        return AlphaSignal(
            alpha_name=self.name,
            timestamp=latest.index[0],
            direction=direction,
            strength=score,
            confidence=confidence,
            features_used=("depth_zscore", "spread_zscore"),
        )

    def evaluate(self, predictions: pd.Series, actuals: pd.Series) -> AlphaMetrics:
        """Evaluate out-of-sample performance."""
        signals = predictions.copy()
        signals[predictions > 0.5] = 1
        signals[predictions < -0.5] = -1
        signals[predictions.abs() <= 0.5] = 0

        metrics = AlphaMetrics.from_predictions(signals, actuals)
        self.cache_metrics(metrics)
        return metrics
