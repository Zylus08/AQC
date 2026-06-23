"""
aqc/execution/slippage_model.py
=================================
Models order slippage based on fixed bps, volatility, spread, and volume.

Author: Saksham Mishra — AlgoQuant Club
"""
from __future__ import annotations
import logging
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

class SlippageModel:
    """Models execution slippage with spread, volatility, and participation rate."""

    def __init__(self, fixed_bps: float = 0.0, volatility_multiplier: float = 0.0):
        self.fixed_bps = fixed_bps
        self.volatility_multiplier = volatility_multiplier

    def estimate_slippage(
        self, 
        price: float, 
        spread: float = 0.0, 
        daily_volatility: float = 0.0,
        trade_size: float = 0.0,
        adv: float = 1e6
    ) -> float:
        """Estimate slippage in absolute price terms incorporating spread and participation rate."""
        slippage = 0.0

        # Fixed component
        if self.fixed_bps > 0:
            slippage += price * (self.fixed_bps / 10000.0)

        # Spread component (cross half the spread)
        if spread > 0:
            slippage += spread / 2.0

        # Volatility component
        if self.volatility_multiplier > 0 and daily_volatility > 0:
            slippage += price * daily_volatility * self.volatility_multiplier
            
        # Liquidity/Participation component (heuristic)
        if trade_size > 0 and adv > 0:
            participation_rate = trade_size / adv
            # If we are more than 1% of ADV, slippage starts increasing exponentially
            if participation_rate > 0.01:
                slippage += price * (participation_rate ** 1.5) * 0.1

        return float(slippage)

    def apply_slippage(
        self, price: float, side: str, spread: float = 0.0, daily_volatility: float = 0.0,
        trade_size: float = 0.0, adv: float = 1e6
    ) -> float:
        slip = self.estimate_slippage(price, spread, daily_volatility, trade_size, adv)
        if side == "BUY":
            return price + slip
        elif side == "SELL":
            return price - slip
        return price
