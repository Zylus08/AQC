"""
tests/test_events.py
====================
Unit tests for the AQC event system.

Tests cover:
* Event construction and validation
* EventQueue thread-safety and stats
* Invalid event rejection
"""

from __future__ import annotations

import threading
import pytest

from aqc.backtester.event import (
    EventType,
    FillEvent,
    FillStatus,
    MarketEvent,
    OrderEvent,
    OrderSide,
    OrderType,
    SignalDirection,
    SignalEvent,
)
from aqc.backtester.event_queue import EventQueue


# ---------------------------------------------------------------------------
# MarketEvent tests
# ---------------------------------------------------------------------------


class TestMarketEvent:
    def test_construction_sets_event_type(self):
        event = MarketEvent(symbol="AAPL", close_price=150.0)
        assert event.event_type == EventType.MARKET

    def test_fields_accessible(self):
        event = MarketEvent(
            symbol="TSLA",
            open_price=200.0,
            high_price=210.0,
            low_price=195.0,
            close_price=205.0,
            volume=5_000_000.0,
        )
        assert event.symbol == "TSLA"
        assert event.close_price == 205.0
        assert event.volume == 5_000_000.0

    def test_is_frozen(self):
        event = MarketEvent(symbol="AAPL", close_price=100.0)
        with pytest.raises((AttributeError, TypeError)):
            event.close_price = 999.0  # type: ignore[misc]

    def test_unique_event_ids(self):
        e1 = MarketEvent(symbol="X")
        e2 = MarketEvent(symbol="X")
        assert e1.event_id != e2.event_id


# ---------------------------------------------------------------------------
# SignalEvent tests
# ---------------------------------------------------------------------------


class TestSignalEvent:
    def test_construction(self):
        sig = SignalEvent(
            symbol="AAPL",
            strategy_id="test",
            direction=SignalDirection.LONG,
            strength=0.8,
        )
        assert sig.event_type == EventType.SIGNAL
        assert sig.direction == SignalDirection.LONG
        assert sig.strength == 0.8

    def test_strength_out_of_range_raises(self):
        with pytest.raises(ValueError, match="strength"):
            SignalEvent(symbol="X", strength=1.5)

    def test_negative_strength_valid(self):
        sig = SignalEvent(symbol="X", strength=-1.0)
        assert sig.strength == -1.0

    def test_metadata_default_empty_dict(self):
        sig = SignalEvent(symbol="X")
        assert sig.metadata == {}


# ---------------------------------------------------------------------------
# OrderEvent tests
# ---------------------------------------------------------------------------


class TestOrderEvent:
    def test_construction(self):
        order = OrderEvent(
            symbol="MSFT",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=100.0,
        )
        assert order.event_type == EventType.ORDER
        assert order.quantity == 100.0

    def test_zero_quantity_raises(self):
        with pytest.raises(ValueError, match="quantity"):
            OrderEvent(symbol="X", quantity=0.0)

    def test_negative_quantity_raises(self):
        with pytest.raises(ValueError, match="quantity"):
            OrderEvent(symbol="X", quantity=-50.0)

    def test_limit_order_without_price_raises(self):
        with pytest.raises(ValueError, match="limit_price"):
            OrderEvent(
                symbol="X",
                order_type=OrderType.LIMIT,
                quantity=100.0,
            )

    def test_limit_order_with_price_valid(self):
        order = OrderEvent(
            symbol="X",
            order_type=OrderType.LIMIT,
            quantity=100.0,
            limit_price=150.0,
        )
        assert order.limit_price == 150.0


# ---------------------------------------------------------------------------
# FillEvent tests
# ---------------------------------------------------------------------------


class TestFillEvent:
    def test_gross_value(self):
        fill = FillEvent(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100.0,
            fill_price=150.0,
        )
        assert fill.gross_value == pytest.approx(15_000.0)

    def test_net_value_buy(self):
        fill = FillEvent(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100.0,
            fill_price=150.0,
            commission=15.0,
        )
        # BUY: cash out = 100 * 150 + 15 commission = 15015
        assert fill.net_value == pytest.approx(15_015.0)

    def test_net_value_sell(self):
        fill = FillEvent(
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=100.0,
            fill_price=160.0,
            commission=16.0,
        )
        # SELL: cash in = -(100 * 160) + 16 = -15984
        assert fill.net_value == pytest.approx(-15_984.0)


# ---------------------------------------------------------------------------
# EventQueue tests
# ---------------------------------------------------------------------------


class TestEventQueue:
    def test_put_and_get(self):
        eq = EventQueue()
        event = MarketEvent(symbol="AAPL")
        eq.put(event)
        retrieved = eq.get(block=False)
        assert retrieved is not None
        assert retrieved.event_id == event.event_id

    def test_empty_queue_returns_none(self):
        eq = EventQueue()
        assert eq.get(block=False) is None

    def test_stats_tracking(self):
        eq = EventQueue()
        eq.put(MarketEvent(symbol="A"))
        eq.put(MarketEvent(symbol="B"))
        eq.get(block=False)
        stats = eq.stats
        assert stats["total_enqueued"] == 2
        assert stats["total_dequeued"] == 1
        assert stats["current_depth"] == 1

    def test_rejects_non_event(self):
        eq = EventQueue()
        with pytest.raises(TypeError):
            eq.put("not an event")  # type: ignore[arg-type]

    def test_flush(self):
        eq = EventQueue()
        for _ in range(5):
            eq.put(MarketEvent(symbol="X"))
        discarded = eq.flush()
        assert discarded == 5
        assert eq.empty()

    def test_thread_safe_concurrent_puts(self):
        """Ensure concurrent puts from multiple threads don't corrupt the queue."""
        eq = EventQueue()
        n_threads = 10
        n_events_per_thread = 100

        def put_events():
            for _ in range(n_events_per_thread):
                eq.put(MarketEvent(symbol="T"))

        threads = [threading.Thread(target=put_events) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert eq.qsize() == n_threads * n_events_per_thread

    def test_priority_put(self):
        """Priority event should be dequeued first."""
        eq = EventQueue()
        normal = MarketEvent(symbol="NORMAL")
        priority = MarketEvent(symbol="PRIORITY")
        eq.put(normal)
        eq.put_priority(priority)
        first = eq.get(block=False)
        assert first is not None
        assert first.symbol == "PRIORITY"
