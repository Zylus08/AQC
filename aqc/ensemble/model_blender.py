"""
aqc/ensemble/model_blender.py
===============================
Blending methods for ensemble alpha models.

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from aqc.alpha.alpha_base import AlphaBase, AlphaMetrics

logger = logging.getLogger(__name__)


class BlendingMethod(Enum):
    """Methods for combining alpha scores."""

    EQUAL_WEIGHT = "EQUAL_WEIGHT"
    INVERSE_VOLATILITY = "INVERSE_VOLATILITY"
    INFORMATION_RATIO = "INFORMATION_RATIO"
    BAYESIAN = "BAYESIAN"


class ModelBlender:
    """Blends predictions from multiple alphas.

    Parameters
    ----------
    method:
        Blending method.
    """

    def __init__(self, method: BlendingMethod = BlendingMethod.EQUAL_WEIGHT) -> None:
        self.method = method
        self._weights: dict[str, float] = {}

    def fit_weights(self, alphas: list[AlphaBase]) -> None:
        """Compute static weights for a list of alphas based on their cached metrics.

        Parameters
        ----------
        alphas:
            List of evaluated alphas.
        """
        weights = {}

        if self.method == BlendingMethod.EQUAL_WEIGHT:
            w = 1.0 / len(alphas) if alphas else 0.0
            weights = {a.name: w for a in alphas}

        elif self.method == BlendingMethod.INFORMATION_RATIO:
            total_ir = 0.0
            for a in alphas:
                ir = max(0.0, a.cached_metrics.information_ratio if a.cached_metrics else 0.0)
                weights[a.name] = ir
                total_ir += ir
            
            if total_ir > 0:
                weights = {k: v / total_ir for k, v in weights.items()}
            else:
                w = 1.0 / len(alphas) if alphas else 0.0
                weights = {a.name: w for a in alphas}

        elif self.method == BlendingMethod.INVERSE_VOLATILITY:
            # We use max_drawdown as a proxy for risk if vol isn't directly available in AlphaMetrics
            total_inv_risk = 0.0
            for a in alphas:
                dd = max(1e-4, a.cached_metrics.max_drawdown_pct if a.cached_metrics else 10.0)
                inv_risk = 1.0 / dd
                weights[a.name] = inv_risk
                total_inv_risk += inv_risk
            
            if total_inv_risk > 0:
                weights = {k: v / total_inv_risk for k, v in weights.items()}

        elif self.method == BlendingMethod.BAYESIAN:
            # Simplified Bayesian: weight ~ exp(Sharpe)
            total_exp = 0.0
            for a in alphas:
                sr = a.cached_metrics.sharpe_ratio if a.cached_metrics else 0.0
                exp_sr = np.exp(max(-3.0, min(3.0, sr))) # Cap exponent to prevent overflow
                weights[a.name] = exp_sr
                total_exp += exp_sr
                
            if total_exp > 0:
                weights = {k: v / total_exp for k, v in weights.items()}

        self._weights = weights
        logger.info("Fitted blending weights: %s", self._weights)

    def blend(self, predictions: dict[str, pd.Series]) -> pd.Series:
        """Blend multiple prediction series into one using fitted weights.

        Parameters
        ----------
        predictions:
            Dictionary mapping alpha_name -> pd.Series of predictions.

        Returns
        -------
        pd.Series
            Composite prediction.
        """
        if not self._weights:
            logger.warning("Weights not fitted. Using equal weights.")
            names = list(predictions.keys())
            self._weights = {name: 1.0 / len(names) for name in names}

        # Align series to common index
        df = pd.DataFrame(predictions).fillna(0.0)
        composite = pd.Series(0.0, index=df.index)

        for name, series in df.items():
            w = self._weights.get(str(name), 0.0)
            composite += w * series

        return composite.clip(-1.0, 1.0)

    @property
    def weights(self) -> dict[str, float]:
        return dict(self._weights)

    def set_weights(self, weights: dict[str, float]) -> None:
        """Manually override weights."""
        total = sum(weights.values())
        if total > 0:
            self._weights = {k: v / total for k, v in weights.items()}
        else:
            self._weights = weights
