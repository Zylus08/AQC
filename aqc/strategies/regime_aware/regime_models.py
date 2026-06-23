"""
aqc/strategies/regime_aware/regime_models.py
==============================================
Regime-conditional model training and inference.

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging
from typing import Optional, Any

import pandas as pd

from aqc.alpha.alpha_base import AlphaBase
from aqc.alpha.alpha_registry import AlphaRegistry
from aqc.regimes.regime_engine import RegimeEngine

logger = logging.getLogger(__name__)


class RegimeConditionalModel:
    """Trains and manages separate models for different regimes.

    Parameters
    ----------
    alpha_class_name:
        The underlying alpha to instantiate per regime.
    regime_engine:
        AQC RegimeEngine instance.
    regime_type:
        Which regime to split on ("volatility", "trend", "hmm").
    **kwargs:
        Params passed to the underlying alpha constructor.
    """

    def __init__(
        self,
        alpha_class_name: str,
        regime_engine: Optional[RegimeEngine] = None,
        regime_type: str = "volatility",
        **kwargs: Any,
    ) -> None:
        self.alpha_class_name = alpha_class_name
        self.regime_engine = regime_engine or RegimeEngine()
        self.regime_type = regime_type
        self.kwargs = kwargs

        self.models: dict[str, AlphaBase] = {}
        self._is_fitted = False

    def fit(self, train_data: pd.DataFrame) -> None:
        """Train separate models per regime."""
        # 1. Detect regimes for the entire history
        snapshots = self.regime_engine.detect_all(train_data)
        
        # 2. Add regime label to train_data
        regime_labels = []
        for _, snap in snapshots.items():
            if self.regime_type == "volatility":
                regime_labels.append(snap.volatility.value)
            elif self.regime_type == "trend":
                regime_labels.append(snap.trend.value)
            else:
                regime_labels.append(snap.hmm)
                
        train_with_regime = train_data.copy()
        # Ensure alignment (assuming snapshots are generated per row)
        if len(regime_labels) == len(train_data):
            train_with_regime["_regime"] = regime_labels
        else:
            # Reindex if snapshots are sparse
            snap_df = pd.DataFrame(index=snapshots.keys())
            if self.regime_type == "volatility":
                snap_df["_regime"] = [s.volatility.value for s in snapshots.values()]
            elif self.regime_type == "trend":
                snap_df["_regime"] = [s.trend.value for s in snapshots.values()]
            else:
                snap_df["_regime"] = [s.hmm for s in snapshots.values()]
                
            train_with_regime = train_data.join(snap_df, how="left").ffill()

        # 3. Train a model for each unique regime
        unique_regimes = train_with_regime["_regime"].dropna().unique()
        alpha_cls = AlphaRegistry.get(self.alpha_class_name)
        
        for regime in unique_regimes:
            logger.info("Training %s for regime: %s", self.alpha_class_name, regime)
            subset = train_with_regime[train_with_regime["_regime"] == regime].drop(columns=["_regime"])
            if len(subset) > 50: # Minimum sample size
                model = alpha_cls(name=f"{self.alpha_class_name}_{regime}", **self.kwargs)
                model.fit(subset)
                self.models[str(regime)] = model
            else:
                logger.warning("Insufficient data for regime %s", regime)

        self._is_fitted = True

    def predict(self, features: pd.DataFrame, current_regime: str) -> pd.Series:
        """Predict using the model corresponding to the current regime."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted.")
            
        model = self.models.get(str(current_regime))
        if model is None:
            # Fallback to the model with the most data (first trained usually, or random)
            if self.models:
                fallback_regime = list(self.models.keys())[0]
                logger.debug("No model for regime %s, falling back to %s", current_regime, fallback_regime)
                model = self.models[fallback_regime]
            else:
                return pd.Series(0.0, index=features.index)
                
        return model.predict(features)
