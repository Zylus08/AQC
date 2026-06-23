"""
aqc/strategies/regime_aware/regime_alpha.py
=============================================
Regime-Aware Alpha implementation.

Dynamically switches between different sub-alphas or modulates
their signals based on the current market regime detected by RegimeEngine.

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
from aqc.alpha.alpha_registry import AlphaRegistry
from aqc.regimes.regime_engine import RegimeEngine

logger = logging.getLogger(__name__)


@register_alpha(name="RegimeAwareAlpha", category=AlphaCategory.REGIME)
class RegimeAwareAlpha(AlphaBase):
    """Switches active alpha logic based on market regime.

    Parameters
    ----------
    name:
        Alpha identifier.
    regime_alpha_map:
        Dictionary mapping regime state strings to alpha class names.
        e.g., {"LOW": "MeanReversionAlpha", "HIGH": "MomentumAlpha"}
    regime_type:
        Which regime to condition on ("volatility", "trend", "hmm").
    """

    def __init__(
        self,
        name: str = "regime_aware_alpha",
        version: str = "1.0.0",
        regime_alpha_map: Optional[dict[str, str]] = None,
        regime_type: str = "volatility",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, version=version)
        
        # Default mapping if none provided
        self.regime_alpha_map = regime_alpha_map or {
            "LOW": "MicropriceAlpha",      # Mean reverting in low vol
            "NORMAL": "OrderFlowAlpha",    # Flow works in normal vol
            "HIGH": "LiquidityAlpha",      # Defensive/breakout in high vol
        }
        self.regime_type = regime_type
        self.kwargs = kwargs

        self._regime_engine = RegimeEngine()
        self._sub_alphas: dict[str, AlphaBase] = {}
        self._instantiate_sub_alphas()

    def _instantiate_sub_alphas(self):
        """Instantiate the alphas defined in the map."""
        for state, alpha_name in self.regime_alpha_map.items():
            cls = AlphaRegistry.get(alpha_name)
            # Pass kwargs only if they match, or let factory handle it. 
            # For simplicity, we assume default constructors are valid.
            try:
                self._sub_alphas[state] = cls(name=f"{self.name}_{state}_{alpha_name}")
            except TypeError as e:
                logger.warning("Failed to instantiate %s for state %s: %s", alpha_name, state, e)

    def metadata(self) -> AlphaMetadata:
        return AlphaMetadata(
            name=self.name,
            version=self.version,
            author="AQC Team",
            category=AlphaCategory.REGIME,
            frequency=AlphaFrequency.TICK,
            description="Dynamically switches sub-alphas based on market regime.",
            parameters={
                "regime_alpha_map": self.regime_alpha_map,
                "regime_type": self.regime_type,
            },
        )

    def fit(self, train_data: pd.DataFrame) -> None:
        """Fit all sub-alphas on the full dataset. 
        
        Alternatively, could use `RegimeConditionalModel` to only fit 
        on data corresponding to that regime.
        """
        for state, alpha in self._sub_alphas.items():
            if hasattr(alpha, "fit") and not alpha.is_fitted:
                logger.info("Fitting sub-alpha %s for state %s", alpha.name, state)
                try:
                    alpha.fit(train_data)
                except Exception as e:
                    logger.error("Failed to fit sub-alpha %s: %s", alpha.name, e)
        self._is_fitted = True

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Predict is complex because features might be alpha-specific.
        
        For RegimeAwareAlpha, we prefer generating signals dynamically.
        If predict is called, we fallback to a default or 0.
        """
        return pd.Series(0.0, index=features.index)

    def generate_signal(self, data: pd.DataFrame) -> AlphaSignal:
        """Detect regime and route to the correct sub-alpha."""
        # 1. Detect regime
        snapshot = self._regime_engine.detect(data)
        
        if self.regime_type == "volatility":
            current_state = snapshot.volatility.value
        elif self.regime_type == "trend":
            current_state = snapshot.trend.value
        else:
            current_state = str(snapshot.hmm)

        # 2. Get correct alpha
        active_alpha = self._sub_alphas.get(current_state)
        
        if active_alpha is None:
            # Fallback
            return AlphaSignal(
                alpha_name=self.name, 
                direction=0,
                metadata={"regime": current_state, "error": "No mapped alpha"}
            )

        # 3. Generate signal
        sub_signal = active_alpha.generate_signal(data)

        # 4. Wrap and return
        return AlphaSignal(
            alpha_name=self.name,
            timestamp=sub_signal.timestamp,
            direction=sub_signal.direction,
            strength=sub_signal.strength,
            confidence=sub_signal.confidence,
            features_used=sub_signal.features_used,
            metadata={
                "regime": current_state,
                "active_alpha": active_alpha.name,
                **sub_signal.metadata
            }
        )

    def evaluate(self, predictions: pd.Series, actuals: pd.Series) -> AlphaMetrics:
        """Evaluate out-of-sample performance."""
        # If we had continuous signals recorded we would evaluate them
        # For RegimeAware, it's better evaluated via tournament/backtest
        metrics = AlphaMetrics.from_predictions(predictions, actuals)
        self.cache_metrics(metrics)
        return metrics
