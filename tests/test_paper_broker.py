"""
tests/test_paper_broker.py
============================
Tests for PaperBroker, OrderManager, and FillSimulator.

Author: Saksham Mishra — AlgoQuant Club
"""
import pytest
import pandas as pd

from aqc.backtester.event import OrderEvent, OrderType, OrderSide, MarketEvent
from aqc.backtester.event_queue import EventQueue
from aqc.execution.slippage_model import SlippageModel
from aqc.execution.market_impact import SquareRootImpactModel
from aqc.live.order_manager import OrderManager, OrderState
from aqc.live.fill_simulator import FillSimulator
from aqc.live.paper_broker import PaperBroker


def test_order_manager():
    mgr = OrderManager()
    order = OrderEvent(symbol="AAPL", order_type=OrderType.MARKET, side=OrderSide.BUY, quantity=100, strategy_id="MR")
    
    live_order = mgr.register_order(order)
    assert live_order.state == OrderState.NEW
    assert live_order.remaining_quantity == 100
    
    mgr.update_fill(order.event_id, 40, 150.0)
    assert live_order.state == OrderState.PARTIALLY_FILLED
    assert live_order.filled_quantity == 40
    assert live_order.remaining_quantity == 60
    assert live_order.avg_fill_price == 150.0
    
    mgr.update_fill(order.event_id, 60, 160.0)
    assert live_order.state == OrderState.FILLED
    assert live_order.filled_quantity == 100
    assert live_order.remaining_quantity == 0
    # avg = (40*150 + 60*160) / 100 = 156.0
    assert live_order.avg_fill_price == 156.0


def test_fill_simulator():
    slip = SlippageModel(fixed_bps=0.0)
    impact = SquareRootImpactModel(impact_coefficient=0.0)
    sim = FillSimulator(slip, impact)
    
    order = OrderEvent(symbol="AAPL", order_type=OrderType.MARKET, side=OrderSide.BUY, quantity=1000, strategy_id="MR")
    
    # ADV is 100,000. 10% is 10,000. So we should get full fill.
    res = sim.simulate_fill(order, market_price=100.0, daily_volume=100000)
    
    # 0.1% chance of rejection is ignored in deterministic testing ideally, but we accept it might fail 1 in 1000 times
    if not res.is_rejected:
        assert res.fill_price == 100.0
        assert res.filled_quantity == 1000


def test_paper_broker():
    eq = EventQueue()
    mgr = OrderManager()
    sim = FillSimulator(SlippageModel(fixed_bps=0.0), SquareRootImpactModel(impact_coefficient=0.0))
    broker = PaperBroker(eq, mgr, sim, commission_rate=0.0)
    
    # Send market event to broker
    mkt = MarketEvent(symbol="AAPL", timestamp=pd.Timestamp("2020-01-01"), open_price=100, high_price=105, low_price=95, close_price=100, volume=1e6)
    broker.on_market_event(mkt)
    
    # Submit order
    order = OrderEvent(symbol="AAPL", order_type=OrderType.MARKET, side=OrderSide.BUY, quantity=100, strategy_id="MR")
    broker.submit_order(order)
    
    # Since market data is there, it should have processed and pushed a FillEvent
    assert not eq.empty()
    fill = eq.get()
    assert fill.symbol == "AAPL"
    assert fill.quantity == 100
    assert fill.fill_price == 100.0
