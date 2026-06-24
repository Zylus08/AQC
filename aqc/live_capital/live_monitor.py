"""
aqc/live_capital/live_monitor.py
==================================
Monitors health metrics of live alphas.

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging

import pandas as pd

from aqc.live_capital.kill_switch import KillSwitch
from aqc.deployment.capital.live_pnl_tracker import LivePnLTracker

logger = logging.getLogger(__name__)


class LiveMonitor:
    """Combines PnL tracking and Kill Switch evaluation."""

    def __init__(self, pnl_tracker: LivePnLTracker, kill_switch: KillSwitch) -> None:
        self.pnl_tracker = pnl_tracker
        self.kill_switch = kill_switch

    def process_tick(self, alpha_name: str, tier_name: str, pnl_inr: float, current_drawdown_pct: float, timestamp: pd.Timestamp) -> bool:
        """Process a live tick update.
        
        Returns True if the alpha is still alive, False if killed.
        """
        # 1. Check if already dead
        if self.kill_switch.is_killed(alpha_name):
            return False
            
        # 2. Record PnL
        self.pnl_tracker.record_trade(alpha_name, tier_name, pnl_inr, timestamp)
        
        # 3. Evaluate kill switch
        if self.kill_switch.evaluate(alpha_name, tier_name, current_drawdown_pct):
            return False
            
        return True
