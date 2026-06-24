"""
aqc/live_capital/kill_switch.py
=================================
Automated shutdown engine for live alphas.

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging

from aqc.deployment.capital.capital_tiers import CapitalTierManager

logger = logging.getLogger(__name__)


class KillSwitch:
    """Monitors live performance and triggers shutdown if thresholds are breached."""

    def __init__(self) -> None:
        self.tier_manager = CapitalTierManager()
        self._killed_alphas: set[str] = set()

    def evaluate(self, alpha_name: str, tier_name: str, current_drawdown_pct: float) -> bool:
        """Evaluate if an alpha should be killed.

        Returns
        -------
        bool
            True if the kill switch was triggered, False otherwise.
        """
        if alpha_name in self._killed_alphas:
            return True

        tier = self.tier_manager.get_tier(tier_name)
        
        if current_drawdown_pct > tier.stop_loss_pct:
            logger.critical("KILL SWITCH TRIGGERED for %s in %s! Drawdown %.2f%% > %.2f%% limit.", 
                            alpha_name, tier.name, current_drawdown_pct * 100, tier.stop_loss_pct * 100)
            self._killed_alphas.add(alpha_name)
            return True
            
        return False

    def is_killed(self, alpha_name: str) -> bool:
        return alpha_name in self._killed_alphas
