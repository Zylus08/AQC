"""
aqc/deployment/capital/live_pnl_tracker.py
============================================
Real-time PnL tracking for small capital deployment.

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class LivePnLTracker:
    """Tracks PnL for deployed alphas across tiers."""

    def __init__(self) -> None:
        self._pnl_records: list[dict] = []
        
    def record_trade(self, alpha_name: str, tier: str, pnl_inr: float, timestamp: pd.Timestamp) -> None:
        self._pnl_records.append({
            "alpha_name": alpha_name,
            "tier": tier,
            "pnl": pnl_inr,
            "timestamp": timestamp
        })
        
    def get_summary(self) -> pd.DataFrame:
        if not self._pnl_records:
            return pd.DataFrame()
            
        df = pd.DataFrame(self._pnl_records)
        summary = df.groupby(["alpha_name", "tier"]).agg(
            total_pnl=("pnl", "sum"),
            n_trades=("pnl", "count"),
            win_rate=("pnl", lambda x: (x > 0).mean())
        ).reset_index()
        
        return summary
