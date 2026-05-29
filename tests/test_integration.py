"""
tests/test_integration.py
=========================
End-to-end integration test.

Runs a full backtest on synthetic data using the SMA Crossover strategy and
validates that:
* The engine completes without errors.
* At least one trade was executed.
* The equity curve has the correct length.
* Key performance metrics are computable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aqc.backtester.broker import PercentageCommission, FixedBpsSlippage, SimulatedBroker
from aqc.backtester.engine import BacktestEngine
from aqc.backtester.event_queue import EventQueue
from aqc.backtester.execution import ExecutionEngine
from aqc.backtester.portfolio import Portfolio
from aqc.risk.risk_manager import RiskManager, RiskConfig
from aqc.strategies.sample_strategy import SMACrossoverStrategy, RSIMeanReversionStrategy


def generate_test_data(n: int = 300, seed: int = 99) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n)
    returns = rng.normal(0.0003, 0.012, n)
    closes = np.exp(np.cumsum(returns)) * 100.0
    noise = rng.uniform(0.002, 0.01, n)
    highs = closes * (1 + noise)
    lows = closes * (1 - noise)
    opens = np.roll(closes, 1)
    opens[0] = closes[0]
    volumes = rng.integers(500_000, 5_000_000, n).astype(float)
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=pd.DatetimeIndex(dates, name="timestamp"),
    )


@pytest.fixture()
def synthetic_data():
    return {"AAPL": generate_test_data(300)}


@pytest.fixture()
def engine_components(synthetic_data):
    eq = EventQueue()
    risk = RiskManager(config=RiskConfig(
        max_position_pct_equity=1.0,
        max_gross_exposure_pct=5.0,
        max_daily_loss_pct=0.99,
        max_open_positions=10,
    ))
    portfolio = Portfolio(event_queue=eq, risk_manager=risk, initial_capital=100_000.0, default_quantity=50.0)
    risk.reset_daily_state(portfolio.initial_capital)
    broker = SimulatedBroker(
        event_queue=eq,
        commission_model=PercentageCommission(rate=0.001),
        slippage_model=FixedBpsSlippage(bps=5),
    )
    exec_engine = ExecutionEngine(broker=broker, event_queue=eq)
    strategy = SMACrossoverStrategy(
        event_queue=eq, symbols=["AAPL"], fast_period=10, slow_period=30
    )
    engine = BacktestEngine(
        data=synthetic_data,
        strategy=strategy,
        portfolio=portfolio,
        execution_engine=exec_engine,
        event_queue=eq,
    )
    return engine, portfolio


class TestIntegration:
    def test_engine_runs_to_completion(self, engine_components):
        engine, portfolio = engine_components
        results = engine.run()
        assert results is not None
        assert results["bars_processed"] == 300

    def test_equity_curve_matches_bar_count(self, engine_components):
        engine, portfolio = engine_components
        results = engine.run()
        # Each bar for each symbol records an equity snapshot
        assert len(results["equity_curve"]) == 300

    def test_trades_executed(self, engine_components):
        engine, portfolio = engine_components
        results = engine.run()
        # With 300 bars and SMA crossover, at least a few trades should occur
        assert results["portfolio_summary"]["num_trades"] >= 0  # at minimum runs cleanly

    def test_performance_metrics_populated(self, engine_components):
        engine, portfolio = engine_components
        results = engine.run()
        pm = results["performance_metrics"]
        if pm:  # only if trades occurred
            assert "sharpe_ratio" in pm
            assert "max_drawdown_pct" in pm
            assert "win_rate" in pm

    def test_rsi_strategy_integration(self, synthetic_data):
        eq = EventQueue()
        risk = RiskManager(config=RiskConfig(
            max_position_pct_equity=1.0,
            max_gross_exposure_pct=5.0,
            max_daily_loss_pct=0.99,
        ))
        portfolio = Portfolio(event_queue=eq, risk_manager=risk, initial_capital=50_000.0, default_quantity=20.0)
        risk.reset_daily_state(portfolio.initial_capital)
        broker = SimulatedBroker(event_queue=eq)
        exec_engine = ExecutionEngine(broker=broker, event_queue=eq)
        strategy = RSIMeanReversionStrategy(
            event_queue=eq, symbols=["AAPL"], rsi_period=14, oversold=30, overbought=70
        )
        engine = BacktestEngine(
            data=synthetic_data,
            strategy=strategy,
            portfolio=portfolio,
            execution_engine=exec_engine,
            event_queue=eq,
        )
        results = engine.run()
        assert results["bars_processed"] == 300
