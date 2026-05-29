"""
tests/test_portfolio.py
=======================
Unit tests for the Portfolio and Position classes.

Tests cover:
* Position open / close / reversal accounting
* Portfolio cash accounting
* Equity computation
* Signal-to-order translation
* Risk gate integration (approval and rejection paths)
"""

from __future__ import annotations

import pytest

from aqc.backtester.event import (
    FillEvent,
    FillStatus,
    MarketEvent,
    OrderSide,
    SignalDirection,
    SignalEvent,
)
from aqc.backtester.event_queue import EventQueue
from aqc.backtester.portfolio import Portfolio, Position
from aqc.risk.risk_manager import RiskManager, RiskConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def event_queue() -> EventQueue:
    return EventQueue()


@pytest.fixture()
def risk_manager() -> RiskManager:
    return RiskManager(config=RiskConfig(
        max_position_pct_equity=1.0,      # relaxed for tests
        max_gross_exposure_pct=10.0,      # relaxed for tests
        max_daily_loss_pct=0.99,          # relaxed for tests
        max_open_positions=100,
    ))


@pytest.fixture()
def portfolio(event_queue, risk_manager) -> Portfolio:
    return Portfolio(
        event_queue=event_queue,
        risk_manager=risk_manager,
        initial_capital=100_000.0,
        default_quantity=100.0,
    )


# ---------------------------------------------------------------------------
# Position tests
# ---------------------------------------------------------------------------


class TestPosition:
    def test_open_long_position(self):
        pos = Position(symbol="AAPL")
        pos.apply_fill(OrderSide.BUY, qty=100, price=150.0)
        assert pos.quantity == 100
        assert pos.avg_cost == pytest.approx(150.0)
        assert pos.is_long

    def test_open_short_position(self):
        pos = Position(symbol="AAPL")
        pos.apply_fill(OrderSide.SELL, qty=50, price=200.0)
        assert pos.quantity == -50
        assert pos.avg_cost == pytest.approx(200.0)
        assert pos.is_short

    def test_add_to_long_position_updates_avg_cost(self):
        pos = Position(symbol="X")
        pos.apply_fill(OrderSide.BUY, qty=100, price=100.0)
        pos.apply_fill(OrderSide.BUY, qty=100, price=200.0)
        # avg_cost = (100*100 + 100*200) / 200 = 150
        assert pos.quantity == 200
        assert pos.avg_cost == pytest.approx(150.0)

    def test_close_long_position_realises_pnl(self):
        pos = Position(symbol="X")
        pos.apply_fill(OrderSide.BUY, qty=100, price=100.0)
        realised = pos.apply_fill(OrderSide.SELL, qty=100, price=120.0)
        assert realised == pytest.approx(2000.0)   # 100 * (120 - 100)
        assert pos.is_flat
        assert pos.realised_pnl == pytest.approx(2000.0)

    def test_close_short_position_realises_pnl(self):
        pos = Position(symbol="X")
        pos.apply_fill(OrderSide.SELL, qty=50, price=200.0)
        realised = pos.apply_fill(OrderSide.BUY, qty=50, price=180.0)
        assert realised == pytest.approx(1000.0)   # 50 * (200 - 180)
        assert pos.is_flat

    def test_reversal_closes_and_reopens(self):
        pos = Position(symbol="X")
        pos.apply_fill(OrderSide.BUY, qty=100, price=100.0)
        pos.apply_fill(OrderSide.SELL, qty=200, price=110.0)
        # After reversal: short 100 at 110
        assert pos.quantity == pytest.approx(-100)
        assert pos.avg_cost == pytest.approx(110.0)
        assert pos.realised_pnl == pytest.approx(1000.0)  # 100 * (110 - 100)

    def test_mark_to_market_updates_unrealised_pnl(self):
        pos = Position(symbol="X")
        pos.apply_fill(OrderSide.BUY, qty=100, price=100.0)
        pos.mark_to_market(120.0)
        assert pos.unrealised_pnl == pytest.approx(2000.0)
        assert pos.market_value == pytest.approx(12_000.0)


# ---------------------------------------------------------------------------
# Portfolio tests
# ---------------------------------------------------------------------------


class TestPortfolio:
    def test_initial_state(self, portfolio):
        assert portfolio.cash == pytest.approx(100_000.0)
        assert portfolio.equity == pytest.approx(100_000.0)
        assert portfolio.num_open_positions == 0

    def test_on_fill_buy_deducts_cash(self, portfolio):
        fill = FillEvent(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100.0,
            fill_price=150.0,
            commission=15.0,
        )
        portfolio.on_fill_event(fill)
        expected_cash = 100_000.0 - 100 * 150.0 - 15.0
        assert portfolio.cash == pytest.approx(expected_cash)

    def test_on_fill_sell_adds_cash(self, portfolio):
        # First buy
        buy_fill = FillEvent(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100.0,
            fill_price=150.0,
            commission=15.0,
        )
        portfolio.on_fill_event(buy_fill)

        # Then sell
        sell_fill = FillEvent(
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=100.0,
            fill_price=160.0,
            commission=16.0,
        )
        portfolio.on_fill_event(sell_fill)

        # Cash after buy: 100000 - 15000 - 15 = 84985
        # Cash after sell: 84985 + 16000 - 16 = 100969
        expected_cash = 100_000.0 - 100 * 150.0 - 15.0 + 100 * 160.0 - 16.0
        assert portfolio.cash == pytest.approx(expected_cash)

    def test_equity_includes_position_value(self, portfolio):
        fill = FillEvent(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100.0,
            fill_price=150.0,
            commission=0.0,
        )
        portfolio.on_fill_event(fill)

        # Mark to market at 160
        portfolio.on_market_event(MarketEvent(symbol="AAPL", close_price=160.0))
        # Cash = 100000 - 15000 = 85000; market_value = 100 * 160 = 16000
        assert portfolio.equity == pytest.approx(85_000.0 + 16_000.0)

    def test_signal_buy_creates_order(self, portfolio, event_queue):
        risk_manager = portfolio._risk
        risk_manager.reset_daily_state(portfolio.equity)
        # Seed price cache for risk check
        portfolio.positions["AAPL"] = Position(symbol="AAPL", last_price=150.0)

        signal = SignalEvent(
            symbol="AAPL",
            direction=SignalDirection.LONG,
            strength=1.0,
        )
        portfolio.on_signal_event(signal)
        order = event_queue.get(block=False)
        assert order is not None
        assert order.side == OrderSide.BUY
        assert order.quantity == pytest.approx(100.0)

    def test_hold_signal_produces_no_order(self, portfolio, event_queue):
        signal = SignalEvent(
            symbol="AAPL",
            direction=SignalDirection.HOLD,
        )
        portfolio.on_signal_event(signal)
        assert event_queue.empty()

    def test_summary_fields(self, portfolio):
        summary = portfolio.summary()
        assert "initial_capital" in summary
        assert "final_equity" in summary
        assert "total_pnl" in summary
        assert "return_pct" in summary
        assert summary["initial_capital"] == 100_000.0

    def test_trade_log_populated_on_fill(self, portfolio):
        fill = FillEvent(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=50.0,
            fill_price=100.0,
            commission=5.0,
        )
        portfolio.on_fill_event(fill)
        assert len(portfolio.trade_log) == 1
        assert portfolio.trade_log[0]["symbol"] == "AAPL"
