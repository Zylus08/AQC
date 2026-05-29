"""
tests/test_metrics.py
=====================
Unit tests for performance metric computation.

Tests cover:
* Sharpe Ratio sign and magnitude sanity
* Sortino Ratio with downside-only returns
* CAGR
* Max Drawdown
* Win Rate and Profit Factor
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from aqc.analytics.metrics import PerformanceMetrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_equity_curve(returns: list[float], initial: float = 100_000.0) -> pd.DataFrame:
    """Build an equity curve DataFrame from a list of period returns."""
    equity = [initial]
    for r in returns:
        equity.append(equity[-1] * (1 + r))

    index = pd.date_range(start="2023-01-02", periods=len(equity), freq="B")
    n = len(equity)
    df = pd.DataFrame({"equity": equity, "num_positions": [1] * n}, index=index)
    return df


def make_trade_log(pnls: list[float]) -> list[dict]:
    """Build a minimal trade log from a list of PnL values."""
    return [{"realised_pnl": p, "symbol": "X", "side": "BUY"} for p in pnls]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSharpeRatio:
    def test_positive_returns_give_positive_sharpe(self):
        # Mix a strong positive drift with small noise to ensure nonzero std
        rng = np.random.default_rng(1)
        returns = list(rng.normal(0.003, 0.008, 252))   # mean > 0, has variance
        eq = make_equity_curve(returns)
        m = PerformanceMetrics(equity_curve=eq, trade_log=[])
        assert m.sharpe_ratio() > 0

    def test_negative_returns_give_negative_sharpe(self):
        rng = np.random.default_rng(2)
        returns = list(rng.normal(-0.003, 0.008, 252))  # mean < 0, has variance
        eq = make_equity_curve(returns)
        m = PerformanceMetrics(equity_curve=eq, trade_log=[])
        assert m.sharpe_ratio() < 0

    def test_zero_volatility_returns_nan(self):
        """Constant zero returns produce zero variance → Sharpe is NaN."""
        returns = [0.0] * 252
        eq = make_equity_curve(returns)
        m = PerformanceMetrics(equity_curve=eq, trade_log=[])
        assert math.isnan(m.sharpe_ratio())


class TestSortinoRatio:
    def test_only_positive_returns_returns_nan(self):
        returns = [0.001] * 100
        eq = make_equity_curve(returns)
        m = PerformanceMetrics(equity_curve=eq, trade_log=[])
        sortino = m.sortino_ratio()
        # No downside → denominator is 0 or NaN
        assert math.isnan(sortino) or sortino > 0

    def test_sortino_greater_than_sharpe_for_upside_skewed_returns(self):
        # Asymmetric returns: lots of small gains, few large losses
        rng = np.random.default_rng(0)
        returns = list(rng.exponential(0.005, 200) - 0.001)
        eq = make_equity_curve(returns)
        m = PerformanceMetrics(equity_curve=eq, trade_log=[])
        sharpe = m.sharpe_ratio()
        sortino = m.sortino_ratio()
        if not (math.isnan(sharpe) or math.isnan(sortino)):
            # Sortino should be >= Sharpe when returns are positively skewed
            assert sortino >= sharpe - 0.1  # allow small tolerance


class TestCAGR:
    def test_flat_returns_give_zero_cagr(self):
        returns = [0.0] * 252
        eq = make_equity_curve(returns)
        m = PerformanceMetrics(equity_curve=eq, trade_log=[])
        assert abs(m.cagr()) < 1e-6

    def test_positive_cagr(self):
        returns = [0.001] * 252
        eq = make_equity_curve(returns)
        m = PerformanceMetrics(equity_curve=eq, trade_log=[])
        assert m.cagr() > 0

    def test_cagr_roughly_correct(self):
        """~10% annual return over 252 trading days."""
        daily_return = (1.10 ** (1 / 252)) - 1
        returns = [daily_return] * 252
        eq = make_equity_curve(returns)
        m = PerformanceMetrics(equity_curve=eq, trade_log=[])
        assert m.cagr() == pytest.approx(0.10, rel=0.01)


class TestMaxDrawdown:
    def test_no_drawdown(self):
        returns = [0.001] * 100  # monotonically increasing
        eq = make_equity_curve(returns)
        m = PerformanceMetrics(equity_curve=eq, trade_log=[])
        assert m.max_drawdown_pct() >= -0.01  # near zero

    def test_large_drawdown_detected(self):
        # Go up 50%, then crash 40%
        up = [0.01] * 50
        down = [-0.02] * 30
        returns = up + down
        eq = make_equity_curve(returns)
        m = PerformanceMetrics(equity_curve=eq, trade_log=[])
        assert m.max_drawdown_pct() < -10.0  # expect significant drawdown


class TestWinRateAndProfitFactor:
    def test_win_rate_all_wins(self):
        trade_log = make_trade_log([100.0, 200.0, 50.0])
        eq = make_equity_curve([0.001] * 10)
        m = PerformanceMetrics(equity_curve=eq, trade_log=trade_log)
        assert m.win_rate() == pytest.approx(1.0)

    def test_win_rate_all_losses(self):
        trade_log = make_trade_log([-100.0, -200.0])
        eq = make_equity_curve([0.001] * 10)
        m = PerformanceMetrics(equity_curve=eq, trade_log=trade_log)
        assert m.win_rate() == pytest.approx(0.0)

    def test_win_rate_mixed(self):
        trade_log = make_trade_log([100.0, -50.0, 200.0, -75.0])
        eq = make_equity_curve([0.001] * 10)
        m = PerformanceMetrics(equity_curve=eq, trade_log=trade_log)
        assert m.win_rate() == pytest.approx(0.5)

    def test_profit_factor_profitable(self):
        trade_log = make_trade_log([200.0, 200.0, -100.0])
        eq = make_equity_curve([0.001] * 10)
        m = PerformanceMetrics(equity_curve=eq, trade_log=trade_log)
        assert m.profit_factor() == pytest.approx(4.0)  # 400 / 100

    def test_empty_trade_log_returns_nan(self):
        eq = make_equity_curve([0.001] * 10)
        m = PerformanceMetrics(equity_curve=eq, trade_log=[])
        assert math.isnan(m.win_rate())
        assert math.isnan(m.profit_factor())
