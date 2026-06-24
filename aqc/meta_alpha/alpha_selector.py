"""
aqc/meta_alpha/alpha_selector.py
==================================
Selects the optimal alpha based on current market regime.

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging

import pandas as pd

from aqc.regimes.regime_engine import RegimeEngine
from aqc.alpha.alpha_base import AlphaBase

logger = logging.getLogger(__name__)


class AlphaSelector:
    """Intelligently selects the best alpha to run."""

    def __init__(self, regime_alpha_map: dict[str, str] = None) -> None:
        self.regime_engine = RegimeEngine()
        
        # Default logical mapping if none provided
        self.regime_alpha_map = regime_alpha_map or {
            "LOW": "MicropriceAlpha",     # Mean reverting
            "NORMAL": "OrderFlowAlpha",   # Directional flow
            "HIGH": "LiquidityAlpha",     # Defensive / Breakout
            "EXTREME": "NONE"             # Flat
        }

    def select(self, data: pd.DataFrame, available_alphas: list[AlphaBase]) -> AlphaBase | None:
        """Select the best alpha for the current bar.

        Parameters
        ----------
        data:
            Recent market data to detect regime.
        available_alphas:
            List of instantiated alphas ready to run.

        Returns
        -------
        AlphaBase or None
            The selected alpha, or None if the strategy is to stay flat.
        """
        snapshot = self.regime_engine.detect(data)
        vol_regime = snapshot.volatility.value
        
        target_name = self.regime_alpha_map.get(vol_regime)
        
        if target_name == "NONE":
            logger.info("Regime is %s. Selecting NONE (Flat).", vol_regime)
            return None

        # Find the alpha in the available list
        for alpha in available_alphas:
            if target_name in alpha.name:
                logger.debug("Regime is %s. Selected %s.", vol_regime, alpha.name)
                return alpha
                
        logger.warning("Target alpha %s not found in available alphas. Defaulting to first.", target_name)
        return available_alphas[0] if available_alphas else None
