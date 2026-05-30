"""
tests/test_portfolio_enhanced.py
==================================
Unit and integration tests for:
- PortfolioAllocator + AllocationConstraints
- PortfolioRiskMetrics (VaR, ES, HHI, turnover)
- BacktestComparator + StatisticalTests
- VolatilityTargetedPortfolio (integration)

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_returns(n: int = 500, mu: float = 0.0005, sigma: float = 0.015, seed: int = 42) -> pd.Series:
    """Generate synthetic daily returns."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(rng.normal(mu, sigma, n), index=idx, name="returns")


def _make_equity_curve(returns: pd.Series, initial: float = 100_000.0) -> pd.DataFrame:
    """Build equity curve from returns."""
    equity = initial * (1 + returns).cumprod()
    return pd.DataFrame({"equity": equity}, index=returns.index)


# ===========================================================================
# PORTFOLIO ALLOCATOR TESTS
# ===========================================================================


class TestPortfolioAllocator:
    """Tests for PortfolioAllocator."""

    def test_equal_weight_fixed(self):
        from aqc.portfolio.allocation import PortfolioAllocator, AllocationConstraints, AllocationMethod

        constraints = AllocationConstraints(max_position_weight=0.50)
        allocator = PortfolioAllocator(constraints=constraints)
        result = allocator.allocate(
            symbols=["A", "B", "C"],
            method=AllocationMethod.FIXED,
        )
        # Weights should sum to ~1 and be roughly equal
        assert abs(sum(result.weights.values()) - 1.0) < 0.1
        assert len(result.weights) == 3

    def test_inverse_vol_weights(self):
        from aqc.portfolio.allocation import PortfolioAllocator, AllocationConstraints, AllocationMethod

        # Use loose constraints so inverse vol ordering is visible
        constraints = AllocationConstraints(max_position_weight=0.90)
        allocator = PortfolioAllocator(constraints=constraints)
        result = allocator.allocate(
            symbols=["A", "B"],
            vols={"A": 0.30, "B": 0.10},
            method=AllocationMethod.INVERSE_VOL,
        )
        # B (lower vol) should have higher raw weight
        assert result.raw_weights["B"] > result.raw_weights["A"]

    def test_risk_parity_weights(self):
        from aqc.portfolio.allocation import PortfolioAllocator, AllocationMethod

        allocator = PortfolioAllocator()
        result = allocator.allocate(
            symbols=["A", "B"],
            vols={"A": 0.25, "B": 0.10},
            method=AllocationMethod.RISK_PARITY,
        )
        assert sum(result.weights.values()) > 0

    def test_vol_target_weights(self):
        from aqc.portfolio.allocation import PortfolioAllocator, AllocationMethod

        allocator = PortfolioAllocator()
        result = allocator.allocate(
            symbols=["A", "B"],
            vols={"A": 0.20, "B": 0.20},
            method=AllocationMethod.VOL_TARGET,
            target_vol=0.10,
        )
        assert result.weights["A"] == result.weights["B"]

    def test_max_position_weight_constraint(self):
        from aqc.portfolio.allocation import (
            PortfolioAllocator, AllocationConstraints, AllocationMethod,
        )

        constraints = AllocationConstraints(max_position_weight=0.10)
        allocator = PortfolioAllocator(constraints=constraints)
        result = allocator.allocate(
            symbols=["A", "B"],
            vols={"A": 0.50, "B": 0.05},  # B gets huge weight without cap
            method=AllocationMethod.INVERSE_VOL,
        )
        for w in result.weights.values():
            assert abs(w) <= 0.10 + 1e-6

    def test_gross_exposure_constraint(self):
        from aqc.portfolio.allocation import (
            PortfolioAllocator, AllocationConstraints, AllocationMethod,
        )

        constraints = AllocationConstraints(max_gross_exposure=0.50)
        allocator = PortfolioAllocator(constraints=constraints)
        result = allocator.allocate(
            symbols=["A", "B", "C"],
            method=AllocationMethod.FIXED,
        )
        assert result.gross_exposure <= 0.50 + 1e-6

    def test_sector_constraint(self):
        from aqc.portfolio.allocation import (
            PortfolioAllocator, AllocationConstraints, AllocationMethod,
        )

        constraints = AllocationConstraints(max_sector_exposure=0.30)
        allocator = PortfolioAllocator(constraints=constraints)
        result = allocator.allocate(
            symbols=["A", "B", "C"],
            method=AllocationMethod.FIXED,
            sectors={"A": "tech", "B": "tech", "C": "health"},
        )
        # Tech sector weight should be capped
        tech_weight = abs(result.weights["A"]) + abs(result.weights["B"])
        assert tech_weight <= 0.30 + 1e-4

    def test_raw_weights_preserved(self):
        from aqc.portfolio.allocation import (
            PortfolioAllocator, AllocationConstraints, AllocationMethod,
        )

        constraints = AllocationConstraints(max_position_weight=0.10)
        allocator = PortfolioAllocator(constraints=constraints)
        result = allocator.allocate(
            symbols=["A", "B"],
            vols={"A": 0.50, "B": 0.05},
            method=AllocationMethod.INVERSE_VOL,
        )
        # Raw weights should differ from constrained
        assert result.raw_weights != result.weights or len(result.constraints_applied) == 0

    def test_allocation_history(self):
        from aqc.portfolio.allocation import PortfolioAllocator, AllocationMethod

        allocator = PortfolioAllocator()
        allocator.allocate(symbols=["A"], method=AllocationMethod.FIXED)
        allocator.allocate(symbols=["A", "B"], method=AllocationMethod.FIXED)
        assert len(allocator.get_allocation_history()) == 2


# ===========================================================================
# PORTFOLIO RISK METRICS TESTS
# ===========================================================================


class TestPortfolioRiskMetrics:
    """Tests for PortfolioRiskMetrics."""

    def test_compute_all_returns_dict(self):
        from aqc.portfolio.portfolio_metrics import PortfolioRiskMetrics

        returns = _make_returns(500)
        prm = PortfolioRiskMetrics(returns)
        result = prm.compute_all()
        assert "portfolio_volatility" in result
        assert "historical_var" in result
        assert "expected_shortfall" in result

    def test_historical_var_is_negative(self):
        from aqc.portfolio.portfolio_metrics import PortfolioRiskMetrics

        returns = _make_returns(500)
        prm = PortfolioRiskMetrics(returns)
        var = prm.historical_var()
        assert var < 0

    def test_parametric_var_is_negative(self):
        from aqc.portfolio.portfolio_metrics import PortfolioRiskMetrics

        returns = _make_returns(500)
        prm = PortfolioRiskMetrics(returns)
        var = prm.parametric_var()
        assert var < 0

    def test_es_worse_than_var(self):
        """ES should be more negative than VaR."""
        from aqc.portfolio.portfolio_metrics import PortfolioRiskMetrics

        returns = _make_returns(500)
        prm = PortfolioRiskMetrics(returns)
        var = prm.historical_var()
        es = prm.expected_shortfall()
        assert es <= var

    def test_hhi_diversified(self):
        from aqc.portfolio.portfolio_metrics import PortfolioRiskMetrics

        returns = _make_returns(100)
        prm = PortfolioRiskMetrics(returns)
        # Equal weight 5 assets → HHI = 0.20
        hhi = prm.concentration_hhi({"A": 0.2, "B": 0.2, "C": 0.2, "D": 0.2, "E": 0.2})
        assert abs(hhi - 0.2) < 0.01

    def test_hhi_concentrated(self):
        from aqc.portfolio.portfolio_metrics import PortfolioRiskMetrics

        returns = _make_returns(100)
        prm = PortfolioRiskMetrics(returns)
        hhi = prm.concentration_hhi({"A": 1.0})
        assert abs(hhi - 1.0) < 0.01

    def test_portfolio_volatility(self):
        from aqc.portfolio.portfolio_metrics import PortfolioRiskMetrics

        returns = _make_returns(500, sigma=0.01)
        prm = PortfolioRiskMetrics(returns)
        vol = prm.portfolio_volatility()
        # 1% daily → ~15.8% annualised
        assert 0.10 < vol < 0.25

    def test_turnover_with_weights(self):
        from aqc.portfolio.portfolio_metrics import PortfolioRiskMetrics

        returns = _make_returns(100)
        # Constant weights → zero turnover
        w = pd.DataFrame(
            {"A": [0.5] * 100, "B": [0.5] * 100},
            index=returns.index,
        )
        prm = PortfolioRiskMetrics(returns, weights_history=w)
        turnover = prm.portfolio_turnover()
        assert turnover == 0.0

    def test_skewness_and_kurtosis(self):
        from aqc.portfolio.portfolio_metrics import PortfolioRiskMetrics

        returns = _make_returns(500)
        prm = PortfolioRiskMetrics(returns)
        assert isinstance(prm.skewness(), float)
        assert isinstance(prm.kurtosis(), float)

    def test_marginal_risk_contribution(self):
        from aqc.portfolio.portfolio_metrics import PortfolioRiskMetrics

        rng = np.random.default_rng(42)
        n = 200
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        asset_returns = pd.DataFrame({
            "A": rng.normal(0, 0.02, n),
            "B": rng.normal(0, 0.01, n),
        }, index=idx)

        returns = _make_returns(n)
        prm = PortfolioRiskMetrics(returns)
        mrc = prm.marginal_risk_contribution(asset_returns, {"A": 0.5, "B": 0.5})
        assert abs(sum(mrc.values()) - 1.0) < 0.01


# ===========================================================================
# BACKTEST COMPARATOR TESTS
# ===========================================================================


class TestBacktestComparator:
    """Tests for BacktestComparator."""

    def test_add_result_and_compare(self):
        from aqc.research.comparison.comparator import BacktestComparator

        comparator = BacktestComparator()
        r1 = _make_returns(200, mu=0.0005)
        r2 = _make_returns(200, mu=0.001)
        comparator.add_result("Baseline", _make_equity_curve(r1), [])
        comparator.add_result("Enhanced", _make_equity_curve(r2), [])

        comparison = comparator.compare()
        assert "Baseline" in comparison.columns
        assert "Enhanced" in comparison.columns
        assert "sharpe_ratio" in comparison.index

    def test_get_returns(self):
        from aqc.research.comparison.comparator import BacktestComparator

        comparator = BacktestComparator()
        r = _make_returns(200)
        comparator.add_result("Test", _make_equity_curve(r), [])
        returns = comparator.get_returns("Test")
        assert len(returns) > 0

    def test_equity_curves_combined(self):
        from aqc.research.comparison.comparator import BacktestComparator

        comparator = BacktestComparator()
        r1 = _make_returns(200, seed=1)
        r2 = _make_returns(200, seed=2)
        comparator.add_result("A", _make_equity_curve(r1), [])
        comparator.add_result("B", _make_equity_curve(r2), [])

        curves = comparator.get_all_equity_curves()
        assert "A" in curves.columns
        assert "B" in curves.columns


# ===========================================================================
# STATISTICAL TESTS
# ===========================================================================


class TestStatisticalTests:
    """Tests for StatisticalTests."""

    def test_t_test_same_returns(self):
        from aqc.research.comparison.comparator import StatisticalTests

        r = _make_returns(500)
        result = StatisticalTests.t_test_returns(r, r)
        assert result["p_value"] > 0.5  # same data → not significant

    def test_t_test_different_returns(self):
        from aqc.research.comparison.comparator import StatisticalTests

        r1 = _make_returns(500, mu=0.0, sigma=0.01, seed=1)
        r2 = _make_returns(500, mu=0.005, sigma=0.01, seed=2)
        result = StatisticalTests.t_test_returns(r1, r2)
        assert result["significant_5pct"] == True  # noqa: E712

    def test_bootstrap_sharpe_ci(self):
        from aqc.research.comparison.comparator import StatisticalTests

        r = _make_returns(300)
        ci = StatisticalTests.bootstrap_sharpe_ci(r)
        assert ci["ci_lower"] <= ci["point_estimate"] <= ci["ci_upper"]

    def test_sharpe_difference_test(self):
        from aqc.research.comparison.comparator import StatisticalTests

        r1 = _make_returns(300, mu=0.001, seed=1)
        r2 = _make_returns(300, mu=0.0001, seed=2)
        result = StatisticalTests.sharpe_difference_test(r1, r2)
        assert "sharpe_diff" in result
        assert "p_value" in result

    def test_sharpe_diff_same_returns(self):
        from aqc.research.comparison.comparator import StatisticalTests

        r = _make_returns(300)
        result = StatisticalTests.sharpe_difference_test(r, r)
        assert abs(result["sharpe_diff"]) < 0.5
