"""
aqc/execution/liquidity_model.py
==================================
Models liquidity constraints and calculates volume participation.

Author: Saksham Mishra — AlgoQuant Club
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class LiquidityModel:
    """Models constraints on order size relative to market volume.

    Parameters
    ----------
    max_participation_rate : float
        Maximum allowed participation as a fraction of ADV (e.g., 0.1 for 10%).
    """

    def __init__(self, max_participation_rate: float = 0.10) -> None:
        self.max_participation_rate = max_participation_rate

    def calculate_participation(self, order_qty: float, daily_volume: float) -> float:
        """Calculate participation rate for a given order.

        Parameters
        ----------
        order_qty : float
            Quantity to trade.
        daily_volume : float
            Average Daily Volume (ADV) or current bar volume.

        Returns
        -------
        float
            Participation rate in [0, 1].
        """
        if daily_volume <= 0:
            return 1.0  # Infinite participation if no volume
        return abs(order_qty) / daily_volume

    def constrain_quantity(self, desired_qty: float, daily_volume: float) -> float:
        """Cap order quantity to respect maximum participation rate.

        Parameters
        ----------
        desired_qty : float
            Requested quantity.
        daily_volume : float
            Market volume.

        Returns
        -------
        float
            Allowed quantity.
        """
        max_qty = daily_volume * self.max_participation_rate
        if abs(desired_qty) > max_qty:
            return max_qty if desired_qty > 0 else -max_qty
        return desired_qty
