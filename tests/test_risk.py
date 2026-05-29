"""
tests/test_risk.py
==================
Unit tests for the RiskManager.

Tests cover:
* Position size limit
* Gross exposure limit
* Daily loss limit
* Max open positions limit
* Custom rule hook
"""

from __future__ import annotations

import pytest

from aqc.backtester.event import OrderEvent, OrderSide, OrderType
from aqc.backtester.event_queue import EventQueue
from aqc.backtester.portfolio import Portfolio, Position
from aqc.risk.risk_manager import RiskConfig, RiskManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_order(symbol: str = "AAPL", qty: float = 100.0) -> OrderEvent:
    return OrderEvent(symbol=symbol, order_type=OrderType.MARKET, side=OrderSide.BUY, quantity=qty)


def make_portfolio(initial_capital: float = 100_000.0) -> Portfolio:
    eq = EventQueue()
    rm = RiskManager(config=RiskConfig(
        max_position_pct_equity=1.0,
        max_gross_exposure_pct=10.0,
        max_daily_loss_pct=0.99,
    ))
    port = Portfolio(event_queue=eq, risk_manager=rm, initial_capital=initial_capital)
    # Seed a position price so the risk manager has a reference price
    port.positions["AAPL"] = Position(symbol="AAPL", last_price=150.0)
    return port


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRiskManager:
    def test_passes_by_default(self):
        rm = RiskManager()
        port = make_portfolio()
        rm.reset_daily_state(port.equity)
        order = make_order(qty=10.0)
        approved, reason = rm.validate_order(order, port)
        assert approved, reason

    def test_max_position_size_rejection(self):
        rm = RiskManager(config=RiskConfig(max_position_size=50.0))
        port = make_portfolio()
        rm.reset_daily_state(port.equity)
        order = make_order(qty=100.0)
        approved, reason = rm.validate_order(order, port)
        assert not approved
        assert "max" in reason.lower() or "size" in reason.lower() or "exceed" in reason.lower()

    def test_max_open_positions_rejection(self):
        rm = RiskManager(config=RiskConfig(
            max_open_positions=1,
            max_position_pct_equity=1.0,
            max_gross_exposure_pct=10.0,
            max_daily_loss_pct=0.99,
        ))
        port = make_portfolio()
        rm.reset_daily_state(port.equity)

        # Add one existing open position
        port.positions["EXISTING"] = Position(symbol="EXISTING", quantity=100.0, last_price=10.0)

        # Now try to open a second one (different symbol, not in positions)
        order = OrderEvent(
            symbol="NEWSTOCK",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=10.0,
        )
        approved, reason = rm.validate_order(order, port)
        assert not approved
        assert "position" in reason.lower()

    def test_daily_loss_rejection(self):
        """Simulate a daily loss that exceeds the limit."""
        rm = RiskManager(config=RiskConfig(
            max_daily_loss_pct=0.01,       # 1% loss limit
            max_position_pct_equity=1.0,
            max_gross_exposure_pct=10.0,
        ))
        port = make_portfolio(100_000.0)
        rm.reset_daily_state(100_000.0)  # baseline equity = 100k

        # Simulate equity dropping 5% — mutate cash directly
        port.cash = 95_000.0  # 5% loss exceeds 1% limit

        order = make_order(qty=10.0)
        approved, reason = rm.validate_order(order, port)
        assert not approved
        assert "daily" in reason.lower() or "loss" in reason.lower()

    def test_custom_rules_hook_can_reject(self):
        """Demonstrate that subclass custom rules are called."""

        class AlwaysRejectRisk(RiskManager):
            def _custom_rules(self, order, portfolio):
                return False, "custom rule blocked this order"

        rm = AlwaysRejectRisk()
        port = make_portfolio()
        rm.reset_daily_state(port.equity)
        order = make_order(qty=1.0)
        approved, reason = rm.validate_order(order, port)
        assert not approved
        assert "custom" in reason.lower()
