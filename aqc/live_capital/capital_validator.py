"""
aqc/live_capital/capital_validator.py
=======================================
Validates capital tier constraints for deployed alphas.

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging

from aqc.deployment.capital.capital_tiers import CapitalTierManager, CapitalTier

logger = logging.getLogger(__name__)


class CapitalValidator:
    """Ensures alphas do not exceed their assigned tier limits."""

    def __init__(self) -> None:
        self.tier_manager = CapitalTierManager()
        self._alpha_tiers: dict[str, str] = {}

    def assign_tier(self, alpha_name: str, tier_name: str = "TIER_1") -> None:
        """Assign an alpha to a specific capital tier."""
        tier = self.tier_manager.get_tier(tier_name)
        self._alpha_tiers[alpha_name] = tier_name
        logger.info("Assigned %s to %s", alpha_name, tier.name)

    def validate_trade(self, alpha_name: str, requested_capital: float) -> float:
        """Validate if a requested capital amount is allowed.
        
        Returns the allowed capital amount (clipped to max if necessary).
        """
        tier_name = self._alpha_tiers.get(alpha_name, "TIER_1")
        tier = self.tier_manager.get_tier(tier_name)
        
        if requested_capital > tier.max_capital_inr:
            logger.warning("Trade rejected/clipped: %s requested %.2f but max for %s is %.2f", 
                           alpha_name, requested_capital, tier.name, tier.max_capital_inr)
            return tier.max_capital_inr
            
        return requested_capital
