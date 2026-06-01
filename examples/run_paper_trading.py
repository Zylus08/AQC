"""
examples/run_paper_trading.py
===============================
End-to-End Paper Trading Demonstration.

Sets up the live data feed, signal router, risk manager, portfolio tracker,
paper broker, persistence, and health monitoring.
Runs an async event loop to process simulated live bars.

Author: Saksham Mishra — AlgoQuant Club
"""
import sys
import os
import asyncio
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np

from aqc.utils.logger import setup_logging
from aqc.backtester.event_queue import EventQueue
from aqc.backtester.portfolio import Portfolio
from aqc.risk.risk_manager import RiskManager
from aqc.strategies.sample_strategy import RSIMeanReversionStrategy

from aqc.live.live_data_feed import SimulatedFeed
from aqc.live.signal_router import SignalRouter
from aqc.live.order_manager import OrderManager
from aqc.live.fill_simulator import FillSimulator
from aqc.execution.slippage_model import SlippageModel
from aqc.execution.market_impact import SquareRootImpactModel
from aqc.live.paper_broker import PaperBroker
from aqc.live.portfolio_tracker import PortfolioTracker
from aqc.live.performance_tracker import PerformanceTracker
from aqc.live.health_monitor import HealthMonitor
from aqc.live.alert_manager import AlertManager, AlertLevel
from aqc.live.persistence import PersistenceLayer
from aqc.live.scheduler import LiveScheduler
from aqc.live.live_engine import LiveTradingEngine
from aqc.live.paper_trading_validator import PaperTradingValidator

logger = logging.getLogger(__name__)


def generate_live_data(n_bars=500):
    idx = pd.date_range(pd.Timestamp.utcnow() - pd.Timedelta(minutes=n_bars), periods=n_bars, freq="1T")
    returns = np.random.normal(0, 0.001, n_bars)
    prices = 100 * np.exp(np.cumsum(returns))
    df = pd.DataFrame({
        "close": prices,
        "open": prices * 0.999,
        "high": prices * 1.002,
        "low": prices * 0.998,
        "volume": np.random.randint(1000, 50000, n_bars)
    }, index=idx)
    return {"AAPL": df}


async def main():
    setup_logging()
    logger.info("Initializing AQC Paper Trading Infrastructure...")

    # Data
    data = generate_live_data()
    feed = SimulatedFeed(data, interval_seconds=0.01) # Fast replay
    
    # Core state
    eq = EventQueue()
    risk_manager = RiskManager()
    portfolio = Portfolio(eq, risk_manager, initial_capital=100_000.0)
    
    # Live Wrappers
    tracker = PortfolioTracker(portfolio)
    router = SignalRouter(risk_manager)
    order_mgr = OrderManager()
    
    slip = SlippageModel(fixed_bps=5.0)
    impact = SquareRootImpactModel(impact_coefficient=0.1)
    sim = FillSimulator(slip, impact)
    
    broker = PaperBroker(eq, order_mgr, sim, commission_rate=0.001)
    
    # Strategy
    strategy = RSIMeanReversionStrategy(eq, ["AAPL"])
    
    # Monitoring & Persistence
    perf_tracker = PerformanceTracker()
    health_mon = HealthMonitor()
    alert_mgr = AlertManager()
    db = PersistenceLayer("reports/paper_trading.db")
    
    # Engine
    engine = LiveTradingEngine(
        data_feed=feed,
        strategy=strategy,
        portfolio_tracker=tracker,
        signal_router=router,
        paper_broker=broker,
        event_queue=eq
    )

    # Scheduler tasks
    scheduler = LiveScheduler()

    async def save_snapshots():
        if tracker.snapshots:
            snap = tracker.snapshots[-1].__dict__
            snap["timestamp"] = str(snap["timestamp"])
            db.save_portfolio_snapshot(snap)
            
        metrics = perf_tracker.compute_metrics(
            pd.Series([s.total_equity for s in tracker.snapshots], 
                      index=[s.timestamp for s in tracker.snapshots])
        )
        m_dict = metrics.__dict__.copy()
        m_dict["timestamp"] = str(m_dict["timestamp"])
        db.save_performance(m_dict)
        
        health = health_mon.check_health()
        h_dict = health.__dict__.copy()
        h_dict["timestamp"] = str(h_dict["timestamp"])
        h_dict["state"] = h_dict["state"].value
        db.save_health(h_dict)
        
        db.save_orders(order_mgr.to_dataframe())
        
        # signals df
        if router.audit_log:
            sig_df = pd.DataFrame(router.audit_log)
            sig_df["timestamp"] = sig_df["timestamp"].astype(str)
            db.save_signals(sig_df)

    scheduler.schedule_periodic(0.5, save_snapshots)
    scheduler.start()

    logger.info("Starting live engine...")
    await engine.start()
    
    await scheduler.stop()
    db.close()
    
    logger.info("Paper Trading Run Complete.")
    
    # Validation
    logger.info("Running Research Validation...")
    eq_series = pd.Series([s.total_equity for s in tracker.snapshots], 
                          index=[s.timestamp for s in tracker.snapshots])
                          
    validator = PaperTradingValidator(eq_series, eq_series) # compare against itself for demo
    report = validator.validate()
    validator.save_report(report, "reports/")
    print(report)

if __name__ == "__main__":
    asyncio.run(main())
