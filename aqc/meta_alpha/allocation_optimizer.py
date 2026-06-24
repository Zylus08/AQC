"""
aqc/meta_alpha/allocation_optimizer.py
========================================
Optimizes capital allocation across the meta-alpha suite.

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class AllocationOptimizer:
    """Dynamically shifts capital allocation weights between alphas."""

    def __init__(self, min_weight: float = 0.0, max_weight: float = 1.0) -> None:
        self.min_weight = min_weight
        self.max_weight = max_weight

    def optimize(self, recent_returns: pd.DataFrame, current_regime: str) -> dict[str, float]:
        """Calculate optimal weights for the current period.

        Parameters
        ----------
        recent_returns:
            DataFrame of recent alpha returns (columns = alpha names).
        current_regime:
            The current volatility regime.

        Returns
        -------
        dict[str, float]
            Allocation weights per alpha.
        """
        if recent_returns.empty:
            return {}

        weights = {}
        alphas = recent_returns.columns
        
        # Base weights via risk parity on recent volatility
        vols = recent_returns.std()
        vols = vols.replace(0, 1e-6)
        inv_vols = 1.0 / vols
        
        total_inv_vol = inv_vols.sum()
        for a in alphas:
            weights[a] = inv_vols[a] / total_inv_vol

        # Apply regime overlays
        # Example logic: in High Vol, penalize mean reverting alphas
        for a in alphas:
            if current_regime == "HIGH" and "Microprice" in a:
                weights[a] *= 0.5
            elif current_regime == "LOW" and "Liquidity" in a:
                weights[a] *= 0.5
                
        # Re-normalize and clip
        total = sum(weights.values())
        if total > 0:
            weights = {k: np.clip(v / total, self.min_weight, self.max_weight) for k, v in weights.items()}
            
            # Final normalization after clipping
            final_total = sum(weights.values())
            weights = {k: v / final_total for k, v in weights.items()}
            
        return weights
