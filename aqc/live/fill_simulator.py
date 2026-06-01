"""
aqc/live/fill_simulator.py
============================
Simulates realistic partial fills, slippage, and market impact for PaperBroker.

Author: Saksham Mishra — AlgoQuant Club
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from aqc.backtester.event import OrderEvent, OrderSide
from aqc.execution.slippage_model import SlippageModel
from aqc.execution.market_impact import SquareRootImpactModel

logger = logging.getLogger(__name__)


@dataclass
class FillResult:
    fill_price: float
    filled_quantity: float
    is_rejected: bool = False
    reason: str = ""


class FillSimulator:
    """Calculates simulated execution reality for paper trading.

    Parameters
    ----------
    slippage_model : SlippageModel
    impact_model : SquareRootImpactModel
    fill_delay_bars : int
        Simulate latency by delaying fill by N bars (default 0).
    """

    def __init__(
        self,
        slippage_model: SlippageModel,
        impact_model: SquareRootImpactModel,
        fill_delay_bars: int = 0,
    ) -> None:
        self.slippage = slippage_model
        self.impact = impact_model
        self.fill_delay_bars = fill_delay_bars

    def simulate_fill(
        self,
        order: OrderEvent,
        market_price: float,
        daily_volume: float = 1e6,
        daily_volatility: float = 0.02,
        remaining_qty: Optional[float] = None,
    ) -> FillResult:
        """Simulate exactly what happens when the order touches the market.

        Parameters
        ----------
        order : OrderEvent
        market_price : float
        daily_volume : float
        daily_volatility : float
        remaining_qty : float, optional

        Returns
        -------
        FillResult
        """
        qty_to_fill = remaining_qty if remaining_qty is not None else order.quantity

        if qty_to_fill <= 0:
            return FillResult(fill_price=0.0, filled_quantity=0.0, is_rejected=True, reason="Zero quantity")

        # Cap fill to max 10% of daily volume to simulate partial fills
        max_fill_size = max(daily_volume * 0.10, 1.0)
        actual_fill_qty = min(qty_to_fill, max_fill_size)

        # 1. Base Slippage
        slip_px = self.slippage.apply_slippage(market_price, order.side.value, daily_volatility)
        
        # 2. Market Impact (applied on top of slippage)
        final_px = self.impact.apply_impact(
            price=slip_px,
            side=order.side.value,
            order_qty=actual_fill_qty,
            daily_volume=daily_volume,
            daily_volatility=daily_volatility
        )

        # 3. Simulate random rejections (0.1% chance) to test resilience
        import random
        if random.random() < 0.001:
            return FillResult(0.0, 0.0, is_rejected=True, reason="Simulated Exchange Reject")

        return FillResult(
            fill_price=final_px,
            filled_quantity=actual_fill_qty,
            is_rejected=False
        )
