"""
aqc/strategies/orderbook_imbalance/imbalance_alpha.py
=======================================================
Order Book Imbalance Alpha.

Implementation of AlphaBase that predicts mid-price direction using
order book volume imbalances and a trained ML model.

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging
from typing import Any, Optional

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
from aqc.strategies.orderbook_imbalance.feature_engine import ImbalanceFeatureEngine
from aqc.strategies.orderbook_imbalance.prediction_models import ImbalancePredictionSuite

logger = logging.getLogger(__name__)


@register_alpha(name="OrderBookImbalanceAlpha", category=AlphaCategory.ORDERBOOK)
class OrderBookImbalanceAlpha(AlphaBase):
    """Predicts mid-price direction from L2 order book imbalances.

    Uses `ImbalanceFeatureEngine` to extract features from snapshots
    and `ImbalancePredictionSuite` for ML inference.

    Parameters
    ----------
    name:
        Alpha identifier.
    model_type:
        ML model to use (e.g., "xgboost", "lightgbm").
    n_levels:
        Order book depth levels to consider.
    threshold:
        Absolute prediction score required to generate a non-zero signal.
    **kwargs:
        Passed to the underlying ML model.
    """

    def __init__(
        self,
        name: str = "ob_imbalance",
        version: str = "1.0.0",
        model_type: str = "xgboost",
        n_levels: int = 10,
        threshold: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, version=version)
        self.model_type = model_type
        self.n_levels = n_levels
        self.threshold = threshold
        self.kwargs = kwargs

        self._feature_engine = ImbalanceFeatureEngine(n_levels=n_levels)
        self._model = ImbalancePredictionSuite(model_type=model_type, **kwargs)

    # ------------------------------------------------------------------
    # AlphaBase Interface
    # ------------------------------------------------------------------

    def metadata(self) -> AlphaMetadata:
        return AlphaMetadata(
            name=self.name,
            version=self.version,
            author="AQC Team",
            category=AlphaCategory.ORDERBOOK,
            frequency=AlphaFrequency.TICK,
            description="Predicts mid-price return using L2 volume imbalances and flow.",
            parameters={
                "model_type": self.model_type,
                "n_levels": self.n_levels,
                "threshold": self.threshold,
                **self.kwargs,
            },
        )

    def fit(self, train_data: pd.DataFrame) -> None:
        """Train the underlying ML model.

        Requires train_data to contain raw snapshots and a target column
        named 'target_dir'.
        """
        if "target_dir" not in train_data.columns:
            raise ValueError("train_data must contain 'target_dir' column.")

        features = self._feature_engine.extract(train_data)
        # Drop rows with NaNs caused by diffs
        aligned = pd.concat([features, train_data["target_dir"]], axis=1).dropna()

        X = aligned.drop(columns=["target_dir"])
        y = aligned["target_dir"]

        self._model.fit(X, y)
        super().fit(train_data)

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Predict alpha scores from a pre-extracted feature matrix."""
        return self._model.predict(features)

    def generate_signal(self, data: pd.DataFrame) -> AlphaSignal:
        """Generate a signal from raw snapshot data.

        Parameters
        ----------
        data:
            A DataFrame containing the latest order book snapshots.
            The last row is used to generate the signal.

        Returns
        -------
        AlphaSignal
        """
        if not self.is_fitted:
            logger.warning("Alpha %s is not fitted. Returning flat signal.", self.name)
            return AlphaSignal(alpha_name=self.name, direction=0)

        # Extract features for the latest row
        latest = data.iloc[[-1]]
        features = self._feature_engine.extract(latest)

        score = float(self.predict(features).iloc[0])

        if score > self.threshold:
            direction = 1
        elif score < -self.threshold:
            direction = -1
        else:
            direction = 0

        # Confidence is mapped from the absolute score
        confidence = min(abs(score) / 0.5, 1.0)

        # Track top contributing features if possible
        feats_used = ()
        if hasattr(self._model, "feature_importance"):
            imp = self._model.feature_importance()
            if not imp.empty:
                feats_used = tuple(imp.head(3).index)

        return AlphaSignal(
            alpha_name=self.name,
            timestamp=latest.index[0],
            direction=direction,
            strength=score,
            confidence=confidence,
            features_used=feats_used,
        )

    def evaluate(self, predictions: pd.Series, actuals: pd.Series) -> AlphaMetrics:
        """Evaluate out-of-sample performance."""
        # Convert continuous score to directional signals based on threshold
        signals = predictions.copy()
        signals[predictions > self.threshold] = 1
        signals[predictions < -self.threshold] = -1
        signals[predictions.abs() <= self.threshold] = 0

        metrics = AlphaMetrics.from_predictions(signals, actuals)
        self.cache_metrics(metrics)
        return metrics
