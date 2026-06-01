"""
tests/test_portfolio_tracker.py
=================================
Tests for live PortfolioTracker.

Author: Saksham Mishra — AlgoQuant Club
"""
import pandas as pd
from aqc.backtester.portfolio import Portfolio
from aqc.backtester.event import FillEvent, OrderSide, FillStatus
from aqc.live.portfolio_tracker import PortfolioTracker


def test_portfolio_tracker():
    from aqc.backtester.event_queue import EventQueue
    from aqc.risk.risk_manager import RiskManager
    eq = EventQueue()
    risk_manager = RiskManager()
    portfolio = Portfolio(eq, risk_manager, initial_capital=100_000.0)
    tracker = PortfolioTracker(portfolio)
    
    snap1 = tracker.snapshot()
    assert snap1.total_equity == 100_000.0
    assert snap1.cash == 100_000.0
    
    # Simulate a fill
    fill = FillEvent(symbol="AAPL", exchange="SIM", side=OrderSide.BUY, quantity=100, fill_price=150.0, commission=1.0, status=FillStatus.FILLED, order_ref="REF", strategy_id="MR")
    portfolio.on_fill_event(fill)
    
    snap2 = tracker.snapshot()
    assert snap2.total_equity == 99_999.0  # 100k - 1 commission
    assert snap2.cash == 84_999.0  # 100k - 15k - 1
    assert snap2.gross_exposure == 15000.0
    
    df = tracker.to_dataframe()
    assert len(df) == 2
