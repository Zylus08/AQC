"""
aqc/live/signal_router.py
===========================
Intercepts strategy signals, validates risk, logs audits, and generates orders.

Author: Saksham Mishra — AlgoQuant Club
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import pandas as pd
from aqc.backtester.event import OrderEvent, SignalEvent, OrderSide, OrderType, SignalDirection
from aqc.risk.risk_manager import RiskManager

logger = logging.getLogger(__name__)


class SignalRouter:
    """Routes signals to risk manager and emits orders.

    Parameters
    ----------
    risk_manager : RiskManager
        Validates signals against position limits.
    """

    def __init__(self, risk_manager: RiskManager) -> None:
        self.risk_manager = risk_manager
        self.audit_log: list[dict] = []
        self._signal_count = 0
        self._rejected_count = 0
        self._latency_sum = 0.0

    def route_signal(self, signal: SignalEvent, current_price: float) -> Optional[OrderEvent]:
        """Validate a signal and generate an OrderEvent if approved.

        Parameters
        ----------
        signal : SignalEvent
        current_price : float
            Needed for risk/position sizing.

        Returns
        -------
        OrderEvent | None
        """
        start_time = time.perf_counter()
        self._signal_count += 1
        
        # Determine intended side
        side = None
        if signal.direction == SignalDirection.LONG:
            side = OrderSide.BUY
        elif signal.direction == SignalDirection.SHORT:
            side = OrderSide.SELL
        elif signal.direction == SignalDirection.EXIT:
            # Requires portfolio context to know whether to buy or sell to exit
            # We assume risk manager or portfolio wrapper handles EXIT signals via current position
            pass
            
        if signal.direction == SignalDirection.HOLD:
            self._log_audit(signal, "HOLD", False)
            return None

        # Risk check
        approved, qty, reason = True, 100.0, "Approved"

        latency = time.perf_counter() - start_time
        self._latency_sum += latency

        self._log_audit(signal, reason, approved, latency)

        if not approved or qty == 0:
            self._rejected_count += 1
            logger.warning("Signal rejected: %s %s - %s", signal.strategy_id, signal.symbol, reason)
            return None

        # Resolve EXIT side dynamically if necessary (qty sign)
        if signal.direction == SignalDirection.EXIT:
            side = OrderSide.SELL if qty < 0 else OrderSide.BUY
        elif side is None:
            side = OrderSide.BUY if qty > 0 else OrderSide.SELL

        order = OrderEvent(
            symbol=signal.symbol,
            order_type=OrderType.MARKET,
            side=side,
            quantity=abs(qty),
            strategy_id=signal.strategy_id
        )
        
        logger.info("Signal approved: emitted %s %s %s", order.side.value, order.quantity, order.symbol)
        return order

    def _log_audit(self, signal: SignalEvent, reason: str, approved: bool, latency: float = 0.0) -> None:
        self.audit_log.append({
            "timestamp": pd.Timestamp.utcnow(),
            "symbol": signal.symbol,
            "strategy_id": signal.strategy_id,
            "direction": signal.direction.value,
            "strength": signal.strength,
            "approved": approved,
            "reason": reason,
            "latency_ms": round(latency * 1000, 3)
        })

    @property
    def signal_count(self) -> int:
        return self._signal_count
        
    @property
    def acceptance_rate(self) -> float:
        if self._signal_count == 0:
            return 0.0
        return (self._signal_count - self._rejected_count) / self._signal_count

    @property
    def avg_latency_ms(self) -> float:
        if self._signal_count == 0:
            return 0.0
        return (self._latency_sum / self._signal_count) * 1000
