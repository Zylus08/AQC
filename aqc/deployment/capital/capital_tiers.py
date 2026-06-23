"""
aqc/deployment/capital/capital_tiers.py
=========================================
Configurable Capital Tiers for live deployment.

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CapitalTier:
    """Defines limits and rules for a specific capital tier."""
    name: str
    max_capital_inr: float
    max_positions: int
    stop_loss_pct: float
    requires_approval: bool = False


class CapitalTierManager:
    """Manages configurable capital tiers for small capital deployment.
    
    Using Indian Rupee (INR) as the base currency.
    """

    def __init__(self) -> None:
        self.tiers = {
            "TIER_1": CapitalTier("Tier 1 (Testing)", 5000.0, 1, 0.01),
            "TIER_2": CapitalTier("Tier 2 (Validation)", 10000.0, 2, 0.02),
            "TIER_3": CapitalTier("Tier 3 (Scaling)", 25000.0, 5, 0.02),
            "TIER_4": CapitalTier("Tier 4 (Full)", 50000.0, 10, 0.03, requires_approval=True),
        }

    def get_tier(self, name: str) -> CapitalTier:
        if name not in self.tiers:
            raise KeyError(f"Unknown tier: {name}")
        return self.tiers[name]

    def upgrade_criteria_met(self, tier_name: str, current_pnl: float, n_trades: int) -> bool:
        """Evaluate if an alpha in a tier is ready for promotion."""
        # Simple heuristic: positive PnL and sufficient trades
        if tier_name == "TIER_1":
            return current_pnl > 100.0 and n_trades >= 10
        elif tier_name == "TIER_2":
            return current_pnl > 500.0 and n_trades >= 25
        elif tier_name == "TIER_3":
            return current_pnl > 2000.0 and n_trades >= 50
        return False
