"""
aqc/live/paper_broker.py
==========================
PaperBroker simulates exchange execution with realistic latency and partial fills.

Author: Saksham Mishra — AlgoQuant Club
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import pandas as pd

from aqc.backtester.event import FillEvent, FillStatus, MarketEvent, OrderEvent, OrderType
from aqc.live.order_manager import OrderManager, OrderState

if TYPE_CHECKING:
    from aqc.backtester.event_queue import EventQueue
    from aqc.live.fill_simulator import FillSimulator

logger = logging.getLogger(__name__)


class PaperBroker:
    """Stateful simulated broker for paper trading.

    Unlike the backtest SimulatedBroker, this handles:
    - Asynchronous execution (orders take time to fill)
    - Partial fills
    - Realistic execution costs (impact + slippage) via FillSimulator
    - Order state management

    Parameters
    ----------
    event_queue : EventQueue
        Where to send FillEvents.
    order_manager : OrderManager
        Tracks order state.
    fill_simulator : FillSimulator
        Computes realistic fill prices.
    commission_rate : float
        Flat rate commission (e.g. 0.001 for 0.1%).
    """

    def __init__(
        self,
        event_queue: "EventQueue",
        order_manager: OrderManager,
        fill_simulator: "FillSimulator",
        commission_rate: float = 0.001,
    ) -> None:
        self.eq = event_queue
        self.order_manager = order_manager
        self.fill_simulator = fill_simulator
        self.commission_rate = commission_rate
        self.latest_market_events: dict[str, MarketEvent] = {}

    def on_market_event(self, event: MarketEvent) -> None:
        """Update market context and attempt to process open orders."""
        self.latest_market_events[event.symbol] = event
        self._process_open_orders()

    def submit_order(self, order: OrderEvent) -> None:
        """Receive an order from the signal router."""
        logger.info("PaperBroker received order: %s %s %s", order.side.value, order.quantity, order.symbol)
        live_order = self.order_manager.register_order(order)
        live_order.update_state(OrderState.SUBMITTED)
        
        # Immediate attempt to process if market data is available
        self._process_open_orders()

    def _process_open_orders(self) -> None:
        """Sweep open orders and simulate fills based on latest market data."""
        open_orders = self.order_manager.get_open_orders()
        
        for live_order in open_orders:
            order = live_order.order_event
            mkt_event = self.latest_market_events.get(order.symbol)
            
            if not mkt_event:
                continue

            # Simulate execution constraints
            res = self.fill_simulator.simulate_fill(
                order=order,
                market_price=mkt_event.close_price,
                daily_volume=mkt_event.volume if mkt_event.volume > 0 else 1e6,
                daily_volatility=0.02, # simplified for now; can map from LiveEngine later
                remaining_qty=live_order.remaining_quantity,
            )

            if res.is_rejected:
                live_order.update_state(OrderState.REJECTED, res.reason)
                logger.warning("PaperBroker rejected order %s: %s", order.event_id, res.reason)
                continue

            if res.filled_quantity > 0:
                commission = (res.filled_quantity * res.fill_price) * self.commission_rate
                
                # Update OrderManager
                self.order_manager.update_fill(
                    order_id=order.event_id,
                    filled_qty=res.filled_quantity,
                    fill_price=res.fill_price
                )

                # Emit FillEvent
                fill_event = FillEvent(
                    symbol=order.symbol,
                    exchange="PAPER",
                    side=order.side,
                    quantity=res.filled_quantity,
                    fill_price=res.fill_price,
                    commission=commission,
                    status=FillStatus.FILLED if live_order.state == OrderState.FILLED else FillStatus.PARTIALLY_FILLED,
                    order_ref=order.event_id,
                    strategy_id=order.strategy_id,
                )
                
                logger.info("PaperBroker filled: %s %s %.2f @ %.4f", 
                            fill_event.side.value, fill_event.symbol, 
                            fill_event.quantity, fill_event.fill_price)
                            
                self.eq.put(fill_event)
