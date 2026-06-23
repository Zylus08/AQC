"""
aqc/strategies/orderflow_alpha/orderflow_alpha.py
===================================================
Order Flow Alpha implementation.

Generates directional signals based on net order flow imbalances
and volume-synchronized probability of informed trading (VPIN).

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
from aqc.strategies.orderflow_alpha.orderflow_features import OrderFlowFeatureEngine

logger = logging.getLogger(__name__)


@register_alpha(name="OrderFlowAlpha", category=AlphaCategory.ORDER_FLOW)
class OrderFlowAlpha(AlphaBase):
    """Generates signals based on order flow toxicity and imbalance.

    Parameters
    ----------
    name:
        Alpha identifier.
    vpin_threshold:
        VPIN level considered "toxic" or informed flow.
    imbalance_threshold:
        Flow imbalance required to generate a directional signal.
    """

    def __init__(
        self,
        name: str = "orderflow_alpha",
        version: str = "1.0.0",
        vpin_threshold: float = 0.8,
        imbalance_threshold: float = 0.4,
    ) -> None:
        super().__init__(name=name, version=version)
        self.vpin_threshold = vpin_threshold
        self.imbalance_threshold = imbalance_threshold

        self._feature_engine = OrderFlowFeatureEngine()
        self._is_fitted = True

    def metadata(self) -> AlphaMetadata:
        return AlphaMetadata(
            name=self.name,
            version=self.version,
            author="AQC Team",
            category=AlphaCategory.ORDER_FLOW,
            frequency=AlphaFrequency.TICK,
            description="Predicts directional momentum from aggressive order flow and VPIN.",
            parameters={
                "vpin_threshold": self.vpin_threshold,
                "imbalance_threshold": self.imbalance_threshold,
            },
        )

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Produce continuous score based on flow imbalance scaled by VPIN."""
        imb = features.get("flow_imbalance_smooth", pd.Series(0.0, index=features.index))
        vpin = features.get("vpin", pd.Series(0.0, index=features.index))
        
        # High VPIN amplifies the flow imbalance signal
        vpin_multiplier = (vpin / self.vpin_threshold).clip(lower=0.5, upper=2.0)
        
        score = (imb / self.imbalance_threshold) * vpin_multiplier
        return score.clip(-1.0, 1.0)

    def generate_signal(self, data: pd.DataFrame) -> AlphaSignal:
        """Generate a signal from combined trade/snapshot data.
        
        Note: `data` must contain trade columns (price, volume, direction) 
        and snapshot columns (mid_price) for this alpha.
        """
        if "volume" not in data.columns or "mid_price" not in data.columns:
            logger.warning("OrderFlowAlpha requires trades and snapshots.")
            return AlphaSignal(alpha_name=self.name, direction=0)

        # For live generation, we pass the same df to both inputs
        features = self._feature_engine.extract(data, data)
        if features.empty:
            return AlphaSignal(alpha_name=self.name, direction=0)
            
        latest = features.iloc[[-1]]
        score = float(self.predict(features).iloc[-1])
        
        direction = 0
        if score > 1.0: # (imb > thresh) adjusted by vpin
            direction = 1
        elif score < -1.0:
            direction = -1
            
        # We cap score magnitude to 1 for the signal attribute
        final_score = max(-1.0, min(1.0, score))
        confidence = min(abs(final_score), 1.0)

        return AlphaSignal(
            alpha_name=self.name,
            timestamp=latest.index[0],
            direction=direction,
            strength=final_score,
            confidence=confidence,
            features_used=("flow_imbalance_smooth", "vpin"),
        )

    def evaluate(self, predictions: pd.Series, actuals: pd.Series) -> AlphaMetrics:
        """Evaluate out-of-sample performance."""
        signals = predictions.copy()
        signals[predictions > 0.8] = 1
        signals[predictions < -0.8] = -1
        signals[predictions.abs() <= 0.8] = 0

        metrics = AlphaMetrics.from_predictions(signals, actuals)
        self.cache_metrics(metrics)
        return metrics
