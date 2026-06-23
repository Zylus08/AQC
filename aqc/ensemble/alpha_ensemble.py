"""
aqc/ensemble/alpha_ensemble.py
================================
AlphaEnsemble implementation.

Combines signals from multiple registered alphas into a single
composite AlphaSignal using a specified ModelBlender.

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
from aqc.ensemble.model_blender import ModelBlender, BlendingMethod

logger = logging.getLogger(__name__)


@register_alpha(name="AlphaEnsemble", category=AlphaCategory.ENSEMBLE)
class AlphaEnsemble(AlphaBase):
    """Composite alpha combining multiple underlying signals.

    Parameters
    ----------
    name:
        Identifier.
    alpha_names:
        List of registered alpha names to include in the ensemble.
    blending_method:
        How to weight the alphas.
    """

    def __init__(
        self,
        name: str = "ensemble_alpha",
        version: str = "1.0.0",
        alpha_names: Optional[list[str]] = None,
        blending_method: BlendingMethod = BlendingMethod.INVERSE_VOLATILITY,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, version=version)
        self.alpha_names = alpha_names or []
        self.blending_method = blending_method
        self.kwargs = kwargs

        self._alphas: list[AlphaBase] = []
        self._blender = ModelBlender(method=blending_method)
        self._instantiate_alphas()

    def _instantiate_alphas(self):
        for aname in self.alpha_names:
            try:
                cls = AlphaRegistry.get(aname)
                alpha = cls(name=f"ens_{aname}")
                self._alphas.append(alpha)
            except Exception as e:
                logger.error("Failed to include %s in ensemble: %s", aname, e)

    def metadata(self) -> AlphaMetadata:
        return AlphaMetadata(
            name=self.name,
            version=self.version,
            author="AQC Team",
            category=AlphaCategory.ENSEMBLE,
            frequency=AlphaFrequency.TICK,
            description=f"Ensemble of {len(self.alpha_names)} alphas.",
            parameters={
                "alpha_names": self.alpha_names,
                "blending_method": self.blending_method.value,
            },
        )

    def fit(self, train_data: pd.DataFrame) -> None:
        """Fit all underlying alphas and the blender weights."""
        for alpha in self._alphas:
            if hasattr(alpha, "fit") and not alpha.is_fitted:
                logger.info("Fitting ensemble component: %s", alpha.name)
                try:
                    alpha.fit(train_data)
                except Exception as e:
                    logger.error("Component fit failed: %s", e)
                    
            # Generate train predictions to evaluate and get metrics for blending
            if "target_dir" in train_data.columns:
                try:
                    # Very rough heuristic fit for blending weights
                    if hasattr(alpha, "_feature_engine"):
                        feats = alpha._feature_engine.extract(train_data)
                        preds = alpha.predict(feats)
                        actuals = train_data["target_dir"]
                        # Align
                        aligned = pd.concat([preds, actuals], axis=1).dropna()
                        metrics = alpha.evaluate(aligned.iloc[:, 0], aligned.iloc[:, 1])
                        # Set cached metrics so blender can use them
                        alpha.cache_metrics(metrics)
                except Exception as e:
                    logger.warning("Could not pre-evaluate %s for blending: %s", alpha.name, e)

        self._blender.fit_weights(self._alphas)
        self._is_fitted = True

    def generate_signal(self, data: pd.DataFrame) -> AlphaSignal:
        """Generate composite signal."""
        predictions = {}
        features_used = []
        
        for alpha in self._alphas:
            try:
                # We extract the single latest signal
                sig = alpha.generate_signal(data)
                predictions[alpha.name] = pd.Series([sig.strength])
                features_used.extend(sig.features_used)
            except Exception as e:
                logger.debug("Alpha %s failed to generate signal: %s", alpha.name, e)
                predictions[alpha.name] = pd.Series([0.0])

        if not predictions:
            return AlphaSignal(alpha_name=self.name, direction=0)

        composite_score_series = self._blender.blend(predictions)
        score = float(composite_score_series.iloc[0])
        
        direction = 1 if score > 0.5 else (-1 if score < -0.5 else 0)
        confidence = min(abs(score), 1.0)
        
        return AlphaSignal(
            alpha_name=self.name,
            timestamp=data.index[-1],
            direction=direction,
            strength=score,
            confidence=confidence,
            features_used=tuple(set(features_used)),
            metadata={"weights": self._blender.weights}
        )

    def predict(self, features: pd.DataFrame) -> pd.Series:
        # Ensemble prediction requires features per sub-alpha, which isn't standard.
        # Fallback to 0.
        return pd.Series(0.0, index=features.index)

    def evaluate(self, predictions: pd.Series, actuals: pd.Series) -> AlphaMetrics:
        metrics = AlphaMetrics.from_predictions(predictions, actuals)
        self.cache_metrics(metrics)
        return metrics
