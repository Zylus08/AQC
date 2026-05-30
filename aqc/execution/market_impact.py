"""
aqc/execution/market_impact.py
================================
Models price impact caused by large orders using the square root model.

Author: Saksham Mishra — AlgoQuant Club
"""
from __future__ import annotations

import logging
import math
from typing import Optional

logger = logging.getLogger(__name__)


class SquareRootImpactModel:
    """Square-root market impact model.

    Expected impact = c * daily_volatility * sqrt(order_qty / daily_volume)

    Parameters
    ----------
    impact_coefficient : float
        Calibration constant 'c' (default 0.1).
    """

    def __init__(self, impact_coefficient: float = 0.1) -> None:
        self.c = impact_coefficient

    def estimate_impact_bps(
        self, order_qty: float, daily_volume: float, daily_volatility: float
    ) -> float:
        """Estimate relative impact in basis points.

        Parameters
        ----------
        order_qty : float
            Quantity to trade.
        daily_volume : float
            Average Daily Volume (ADV).
        daily_volatility : float
            Daily volatility (fraction, e.g., 0.02 for 2%).

        Returns
        -------
        float
            Impact in basis points.
        """
        if daily_volume <= 0:
            return 0.0
        
        participation = abs(order_qty) / daily_volume
        impact_fraction = self.c * daily_volatility * math.sqrt(participation)
        return impact_fraction * 10000.0

    def estimate_impact_price(
        self, price: float, order_qty: float, daily_volume: float, daily_volatility: float
    ) -> float:
        """Estimate absolute price impact.

        Parameters
        ----------
        price : float
            Current market price.
        order_qty : float
            Quantity to trade.
        daily_volume : float
            Average Daily Volume (ADV).
        daily_volatility : float
            Daily volatility (fraction).

        Returns
        -------
        float
            Absolute price difference due to impact.
        """
        impact_bps = self.estimate_impact_bps(order_qty, daily_volume, daily_volatility)
        return price * (impact_bps / 10000.0)

    def apply_impact(
        self, price: float, side: str, order_qty: float, daily_volume: float, daily_volatility: float
    ) -> float:
        """Apply impact to an execution price.

        Parameters
        ----------
        price : float
            Base execution price.
        side : str
            "BUY" or "SELL".
        order_qty : float
            Quantity traded.
        daily_volume : float
            Market volume.
        daily_volatility : float
            Daily volatility.

        Returns
        -------
        float
            Impact-adjusted execution price.
        """
        impact = self.estimate_impact_price(price, order_qty, daily_volume, daily_volatility)
        if side == "BUY":
            return price + impact
        elif side == "SELL":
            return price - impact
        return price
