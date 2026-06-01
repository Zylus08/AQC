"""
aqc/live/order_manager.py
===========================
State machine for tracking order lifecycles in live trading.

Author: Saksham Mishra — AlgoQuant Club
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import pandas as pd

from aqc.backtester.event import OrderEvent, OrderSide, OrderType

logger = logging.getLogger(__name__)


class OrderState(Enum):
    NEW = auto()
    SUBMITTED = auto()
    PARTIALLY_FILLED = auto()
    FILLED = auto()
    CANCELLED = auto()
    REJECTED = auto()


@dataclass
class LiveOrder:
    """Tracks the state of a live order."""
    order_event: OrderEvent
    state: OrderState = OrderState.NEW
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    created_at: pd.Timestamp = field(default_factory=lambda: pd.Timestamp.utcnow())
    updated_at: pd.Timestamp = field(default_factory=lambda: pd.Timestamp.utcnow())
    reason: str = ""

    @property
    def remaining_quantity(self) -> float:
        return self.order_event.quantity - self.filled_quantity

    def update_state(self, new_state: OrderState, reason: str = "") -> None:
        self.state = new_state
        self.reason = reason
        self.updated_at = pd.Timestamp.utcnow()


class OrderManager:
    """Manages the pool of active and historical live orders."""

    def __init__(self) -> None:
        self.orders: dict[str, LiveOrder] = {}

    def register_order(self, order: OrderEvent) -> LiveOrder:
        """Register a new order."""
        live_order = LiveOrder(order_event=order)
        self.orders[order.event_id] = live_order
        return live_order

    def get_order(self, order_id: str) -> Optional[LiveOrder]:
        return self.orders.get(order_id)

    def get_open_orders(self) -> list[LiveOrder]:
        """Return all orders that are not terminal."""
        terminal_states = {OrderState.FILLED, OrderState.CANCELLED, OrderState.REJECTED}
        return [o for o in self.orders.values() if o.state not in terminal_states]

    def update_fill(self, order_id: str, filled_qty: float, fill_price: float) -> None:
        """Update an order with a partial or complete fill."""
        order = self.orders.get(order_id)
        if not order:
            logger.warning("Received fill for unknown order %s", order_id)
            return

        total_qty_before = order.filled_quantity
        new_total_qty = total_qty_before + filled_qty
        
        # Calculate new average fill price
        if new_total_qty > 0:
            order.avg_fill_price = ((total_qty_before * order.avg_fill_price) + (filled_qty * fill_price)) / new_total_qty
            
        order.filled_quantity = new_total_qty
        
        if order.filled_quantity >= order.order_event.quantity:
            order.update_state(OrderState.FILLED)
        else:
            order.update_state(OrderState.PARTIALLY_FILLED)

    def to_dataframe(self) -> pd.DataFrame:
        """Export order history for persistence and UI."""
        if not self.orders:
            return pd.DataFrame()
            
        data = []
        for o in self.orders.values():
            data.append({
                "order_id": o.order_event.event_id,
                "symbol": o.order_event.symbol,
                "side": o.order_event.side.value,
                "type": o.order_event.order_type.value,
                "target_qty": o.order_event.quantity,
                "filled_qty": o.filled_quantity,
                "avg_price": o.avg_fill_price,
                "state": o.state.name,
                "strategy": o.order_event.strategy_id,
                "created_at": o.created_at,
                "updated_at": o.updated_at,
            })
        return pd.DataFrame(data)
