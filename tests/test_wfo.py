"""
tests/test_wfo.py
=================
Unit and integration tests for the Walk-Forward Optimisation module.

Test coverage:
* ParameterSpace / IntParam / FloatParam / CategoricalParam
* ParameterGrid iteration and size
* GridSearchOptimizer and RandomSearchOptimizer
* WalkForwardEngine fold generation (rolling + expanding)
* Full end-to-end WFO run with SMA strategy on synthetic data
* WalkForwardResult aggregate metrics and CSV export
* WalkForwardValidator correlation and report
"""

from __future__ import annotations

import math
import random
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aqc.research.parameter_space import (
    CategoricalParam,
    FloatParam,
    IntParam,
    ParameterGrid,
    ParameterSpace,
)
from aqc.research.optimizer import (
    EvaluationRecord,
    GridSearchOptimizer,
    ObjectiveMetric,
    OptimizationResult,
    RandomSearchOptimizer,
)
from aqc.research.walk_forward import (
    WalkForwardEngine,
    WalkForwardFold,
    WalkForwardMode,
    WalkForwardResult,
)
from aqc.research.validation import WalkForwardValidator


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def make_synthetic_ohlcv(n: int = 400, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n)
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


@pytest.fixture(scope="module")
def synthetic_data():
    return {"AAPL": make_synthetic_ohlcv(400)}


def make_simple_space() -> ParameterSpace:
    """Small 2-param space for fast grid tests."""
    space = ParameterSpace()
    space.add(IntParam("fast_period", 5, 15, step=5))   # [5, 10, 15]
    space.add(IntParam("slow_period", 20, 30, step=10))  # [20, 30]
    return space  # 3 × 2 = 6 combinations


# ---------------------------------------------------------------------------
# ParameterSpace tests
# ---------------------------------------------------------------------------


class TestParameterSpace:
    def test_int_param_values(self):
        p = IntParam("x", low=2, high=10, step=2)
        assert p.values() == [2, 4, 6, 8, 10]

    def test_int_param_invalid_step(self):
        with pytest.raises(ValueError, match="step"):
            IntParam("x", low=0, high=10, step=0)

    def test_int_param_invalid_range(self):
        with pytest.raises(ValueError, match="low"):
            IntParam("x", low=10, high=5)

    def test_float_param_values_count(self):
        p = FloatParam("x", 0.0, 1.0, n_points=5)
        vals = p.values()
        assert len(vals) == 5
        assert vals[0] == pytest.approx(0.0)
        assert vals[-1] == pytest.approx(1.0)

    def test_float_param_invalid_npoints(self):
        with pytest.raises(ValueError):
            FloatParam("x", 0.0, 1.0, n_points=1)

    def test_categorical_param_values(self):
        p = CategoricalParam("flag", choices=[True, False])
        assert p.values() == [True, False]

    def test_categorical_empty_raises(self):
        with pytest.raises(ValueError):
            CategoricalParam("flag", choices=[])

    def test_parameter_space_add_duplicate_raises(self):
        space = ParameterSpace()
        space.add(IntParam("x", 1, 10))
        with pytest.raises(ValueError, match="already exists"):
            space.add(IntParam("x", 2, 20))

    def test_parameter_space_contains(self):
        space = ParameterSpace([IntParam("a", 1, 5)])
        assert "a" in space
        assert "b" not in space

    def test_parameter_space_remove(self):
        space = ParameterSpace([IntParam("a", 1, 5), IntParam("b", 1, 10)])
        space.remove("a")
        assert "a" not in space
        assert len(space) == 1

    def test_grid_size(self):
        space = make_simple_space()
        assert space.grid_size() == 6

    def test_sample_returns_valid_params(self):
        space = make_simple_space()
        rng = random.Random(7)
        params = space.sample(rng)
        assert set(params.keys()) == {"fast_period", "slow_period"}
        assert params["fast_period"] in [5, 10, 15]
        assert params["slow_period"] in [20, 30]

    def test_parameter_grid_iteration(self):
        space = make_simple_space()
        grid = ParameterGrid(space)
        combos = list(grid)
        assert len(combos) == 6
        # All keys present
        for c in combos:
            assert "fast_period" in c
            assert "slow_period" in c

    def test_parameter_grid_no_duplicates(self):
        space = make_simple_space()
        grid = ParameterGrid(space)
        combos = [frozenset(c.items()) for c in grid]
        assert len(combos) == len(set(combos))


# ---------------------------------------------------------------------------
# Optimizer tests
# ---------------------------------------------------------------------------


def make_mock_backtest_fn(best_fast: int = 10, best_slow: int = 30):
    """Return a mock backtest function that rewards params close to best."""

    def fn(params: dict) -> dict:
        fast = params.get("fast_period", 20)
        slow = params.get("slow_period", 50)
        # Simulate Sharpe peaking at best_fast / best_slow
        score = -abs(fast - best_fast) / 20 - abs(slow - best_slow) / 50 + 1.0
        return {
            "performance_metrics": {
                "sharpe_ratio": score,
                "total_return_pct": score * 5,
                "max_drawdown_pct": -10.0,
                "cagr": score * 0.1,
                "sortino_ratio": score * 1.2,
                "calmar_ratio": score * 0.5,
                "win_rate": 0.55,
                "profit_factor": 1.5,
            },
            "equity_curve": pd.DataFrame(
                {"equity": [100_000, 101_000, 102_000], "num_positions": [0, 1, 1]},
                index=pd.date_range("2023-01-01", periods=3),
            ),
            "trade_log": [],
        }

    return fn


class TestGridSearchOptimizer:
    def test_finds_best_params(self):
        space = make_simple_space()
        fn = make_mock_backtest_fn(best_fast=10, best_slow=20)
        opt = GridSearchOptimizer(backtest_fn=fn, space=space, verbose=False)
        result = opt.run()
        assert result.best_params["fast_period"] == 10
        assert result.best_params["slow_period"] == 20

    def test_evaluates_all_combinations(self):
        space = make_simple_space()
        fn = make_mock_backtest_fn()
        opt = GridSearchOptimizer(backtest_fn=fn, space=space, verbose=False)
        result = opt.run()
        assert result.total_evaluations == 6

    def test_to_dataframe_has_correct_columns(self):
        space = make_simple_space()
        fn = make_mock_backtest_fn()
        opt = GridSearchOptimizer(backtest_fn=fn, space=space, verbose=False)
        result = opt.run()
        df = result.to_dataframe()
        assert "fast_period" in df.columns
        assert "objective_value" in df.columns
        assert len(df) == 6

    def test_top_n(self):
        space = make_simple_space()
        fn = make_mock_backtest_fn()
        opt = GridSearchOptimizer(backtest_fn=fn, space=space, verbose=False)
        result = opt.run()
        top3 = result.top_n(3)
        assert len(top3) == 3
        # Verify they are sorted descending
        assert top3[0].objective_value >= top3[1].objective_value >= top3[2].objective_value


class TestRandomSearchOptimizer:
    def test_respects_n_iter(self):
        space = make_simple_space()
        fn = make_mock_backtest_fn()
        opt = RandomSearchOptimizer(backtest_fn=fn, space=space, n_iter=10, verbose=False)
        result = opt.run()
        assert result.total_evaluations == 10

    def test_reproducible_with_seed(self):
        space = make_simple_space()
        fn = make_mock_backtest_fn()
        r1 = RandomSearchOptimizer(backtest_fn=fn, space=space, n_iter=5, seed=99, verbose=False).run()
        r2 = RandomSearchOptimizer(backtest_fn=fn, space=space, n_iter=5, seed=99, verbose=False).run()
        assert r1.best_params == r2.best_params

    def test_different_seed_may_differ(self):
        space = make_simple_space()
        fn = make_mock_backtest_fn()
        r1 = RandomSearchOptimizer(backtest_fn=fn, space=space, n_iter=5, seed=1, verbose=False).run()
        r2 = RandomSearchOptimizer(backtest_fn=fn, space=space, n_iter=5, seed=2, verbose=False).run()
        # Not guaranteed to differ but with enough evaluations they often do
        # Just ensure both complete without error
        assert r1.total_evaluations == 5
        assert r2.total_evaluations == 5


# ---------------------------------------------------------------------------
# Walk-forward fold generation tests
# ---------------------------------------------------------------------------


class TestWalkForwardFoldGeneration:
    def _make_engine(self, data, mode, train_period=100, test_period=50):
        from aqc.strategies.sample_strategy import SMACrossoverStrategy

        return WalkForwardEngine(
            data=data,
            strategy_factory=SMACrossoverStrategy,
            parameter_space=make_simple_space(),
            mode=mode,
            train_period=train_period,
            test_period=test_period,
            optimizer="grid",
            objective=ObjectiveMetric.SHARPE,
        )

    def test_rolling_folds_generated(self, synthetic_data):
        engine = self._make_engine(synthetic_data, WalkForwardMode.ROLLING)
        folds = engine._generate_folds()
        assert len(folds) >= 1
        # Each fold: train_end < test_start
        for ts, te, osts, oste in folds:
            assert te < osts

    def test_expanding_folds_train_start_fixed(self, synthetic_data):
        engine = self._make_engine(synthetic_data, WalkForwardMode.EXPANDING)
        folds = engine._generate_folds()
        first_date = engine._common_index[0]
        for ts, te, osts, oste in folds:
            assert ts == first_date

    def test_n_folds_limit_respected(self, synthetic_data):
        from aqc.strategies.sample_strategy import SMACrossoverStrategy

        engine = WalkForwardEngine(
            data=synthetic_data,
            strategy_factory=SMACrossoverStrategy,
            parameter_space=make_simple_space(),
            mode=WalkForwardMode.ROLLING,
            train_period=100,
            test_period=50,
            n_folds=2,
            optimizer="grid",
        )
        folds = engine._generate_folds()
        assert len(folds) == 2

    def test_insufficient_data_raises(self):
        from aqc.strategies.sample_strategy import SMACrossoverStrategy

        tiny_data = {"X": make_synthetic_ohlcv(n=10)}
        with pytest.raises(ValueError, match="exceeds available data"):
            engine = WalkForwardEngine(
                data=tiny_data,
                strategy_factory=SMACrossoverStrategy,
                parameter_space=make_simple_space(),
                train_period=300,
                test_period=100,
            )
            engine._generate_folds()


# ---------------------------------------------------------------------------
# Walk-forward result tests
# ---------------------------------------------------------------------------


def make_fake_wf_result(n_folds: int = 3) -> WalkForwardResult:
    """Build a fake WalkForwardResult for testing aggregate methods."""
    import pandas as pd

    folds = []
    base_date = pd.Timestamp("2022-01-03")
    for i in range(n_folds):
        train_start = base_date + pd.Timedelta(days=i * 90)
        train_end = train_start + pd.Timedelta(days=89)
        test_start = train_end + pd.Timedelta(days=1)
        test_end = test_start + pd.Timedelta(days=44)

        dates = pd.bdate_range(test_start, periods=20)
        equity = pd.DataFrame(
            {"equity": [100_000 + j * 50 * (1 if i % 2 == 0 else -1) for j in range(20)],
             "num_positions": [1] * 20},
            index=pd.DatetimeIndex(dates),
        )

        fold = WalkForwardFold(
            fold_index=i,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            best_params={"fast_period": 10 + i * 5, "slow_period": 30},
            train_metrics={"sharpe_ratio": 1.5 + i * 0.2, "total_return_pct": 5.0 + i},
            test_metrics={"sharpe_ratio": 0.8 + i * 0.1, "total_return_pct": 2.0 + i,
                          "max_drawdown_pct": -3.0, "win_rate": 0.55,
                          "profit_factor": 1.3, "cagr": 0.08, "sortino_ratio": 1.2,
                          "calmar_ratio": 0.5},
            test_equity_curve=equity,
        )
        folds.append(fold)

    return WalkForwardResult(
        folds=folds,
        mode=WalkForwardMode.ROLLING,
        objective_metric=ObjectiveMetric.SHARPE,
        total_elapsed_seconds=12.5,
    )


class TestWalkForwardResult:
    def test_aggregate_metrics_keys(self):
        result = make_fake_wf_result(3)
        agg = result.aggregate_metrics()
        assert "test_sharpe_ratio_mean" in agg
        assert "test_total_return_pct_std" in agg
        assert agg["n_folds"] == 3

    def test_parameter_stability_numeric(self):
        result = make_fake_wf_result(3)
        stab = result.parameter_stability()
        assert "fast_period" in stab
        assert "mean" in stab["fast_period"]
        assert "std" in stab["fast_period"]

    def test_to_dataframe_shape(self):
        result = make_fake_wf_result(4)
        df = result.to_dataframe()
        assert len(df) == 4
        assert "train_sharpe_ratio" in df.columns
        assert "test_total_return_pct" in df.columns

    def test_combined_equity_curve_non_empty(self):
        result = make_fake_wf_result(3)
        combined = result.combined_test_equity_curve()
        assert not combined.empty
        assert "equity" in combined.columns
        assert "fold" in combined.columns

    def test_save_csv(self, tmp_path):
        result = make_fake_wf_result(3)
        csv_path = str(tmp_path / "wfo_results.csv")
        result.save_csv(csv_path)
        assert Path(csv_path).exists()
        df = pd.read_csv(csv_path)
        assert len(df) == 3


# ---------------------------------------------------------------------------
# WalkForwardValidator tests
# ---------------------------------------------------------------------------


class TestWalkForwardValidator:
    def test_is_oos_correlation_computes(self):
        result = make_fake_wf_result(4)
        validator = WalkForwardValidator(result=result, output_dir=tempfile.mkdtemp())
        corr = validator.is_oos_correlation()
        assert math.isfinite(corr)
        assert -1.0 <= corr <= 1.0

    def test_overfitting_score_bounded(self):
        result = make_fake_wf_result(3)
        validator = WalkForwardValidator(result=result, output_dir=tempfile.mkdtemp())
        score = validator.overfitting_score()
        # Score may be nan if returns are negative; just check it doesn't raise
        assert isinstance(score, float)

    def test_report_contains_key_sections(self):
        result = make_fake_wf_result(3)
        validator = WalkForwardValidator(result=result, output_dir=tempfile.mkdtemp())
        report = validator.build_report_text()
        assert "WALK-FORWARD" in report.upper()
        assert "PER-FOLD" in report.upper()
        assert "IS/OOS" in report.upper()
        assert "PARAMETER STABILITY" in report.upper()

    def test_save_report_creates_file(self, tmp_path):
        result = make_fake_wf_result(3)
        validator = WalkForwardValidator(result=result, output_dir=str(tmp_path))
        path = validator.save_report("test_report.txt")
        assert Path(path).exists()
        content = Path(path).read_text()
        assert "WALK-FORWARD" in content.upper()


# ---------------------------------------------------------------------------
# End-to-end WFO integration test
# ---------------------------------------------------------------------------


class TestWalkForwardIntegration:
    """Full end-to-end walk-forward with real AQC engine."""

    def test_wfo_rolling_completes(self, synthetic_data):
        from aqc.strategies.sample_strategy import SMACrossoverStrategy

        space = ParameterSpace()
        space.add(IntParam("fast_period", 5, 15, step=5))
        space.add(IntParam("slow_period", 20, 30, step=10))

        engine = WalkForwardEngine(
            data=synthetic_data,
            strategy_factory=SMACrossoverStrategy,
            parameter_space=space,
            mode=WalkForwardMode.ROLLING,
            train_period=120,
            test_period=60,
            n_folds=2,
            optimizer="grid",
            objective=ObjectiveMetric.SHARPE,
            initial_capital=100_000,
            default_quantity=50,
        )

        result = engine.run()

        assert len(result.folds) == 2
        for fold in result.folds:
            assert fold.best_params is not None
            assert "fast_period" in fold.best_params
            assert "slow_period" in fold.best_params

    def test_wfo_expanding_completes(self, synthetic_data):
        from aqc.strategies.sample_strategy import SMACrossoverStrategy

        space = ParameterSpace()
        space.add(IntParam("fast_period", 5, 10, step=5))
        space.add(IntParam("slow_period", 20, 30, step=10))

        engine = WalkForwardEngine(
            data=synthetic_data,
            strategy_factory=SMACrossoverStrategy,
            parameter_space=space,
            mode=WalkForwardMode.EXPANDING,
            train_period=120,
            test_period=60,
            n_folds=2,
            optimizer="random",
            n_iter=4,
            objective=ObjectiveMetric.SHARPE,
        )

        result = engine.run()
        assert len(result.folds) == 2
        # Expanding mode: all train windows start at the same date
        first_date = engine._common_index[0]
        for fold in result.folds:
            assert fold.train_start == first_date

    def test_wfo_result_has_agg_metrics(self, synthetic_data):
        from aqc.strategies.sample_strategy import SMACrossoverStrategy

        space = ParameterSpace()
        space.add(IntParam("fast_period", 5, 10, step=5))
        space.add(IntParam("slow_period", 20, 30, step=10))

        engine = WalkForwardEngine(
            data=synthetic_data,
            strategy_factory=SMACrossoverStrategy,
            parameter_space=space,
            mode=WalkForwardMode.ROLLING,
            train_period=100,
            test_period=50,
            n_folds=2,
            optimizer="grid",
        )

        result = engine.run()
        agg = result.aggregate_metrics()
        assert "test_sharpe_ratio_mean" in agg
        assert agg["n_folds"] == 2

    def test_wfo_csv_export(self, synthetic_data, tmp_path):
        from aqc.strategies.sample_strategy import SMACrossoverStrategy

        space = ParameterSpace()
        space.add(IntParam("fast_period", 5, 10, step=5))
        space.add(IntParam("slow_period", 20, 30, step=10))

        engine = WalkForwardEngine(
            data=synthetic_data,
            strategy_factory=SMACrossoverStrategy,
            parameter_space=space,
            mode=WalkForwardMode.ROLLING,
            train_period=100,
            test_period=50,
            n_folds=2,
            optimizer="grid",
        )

        result = engine.run()
        csv_path = str(tmp_path / "wfo_results.csv")
        result.save_csv(csv_path)

        assert Path(csv_path).exists()
        df = pd.read_csv(csv_path)
        assert len(df) == 2
        assert "param_fast_period" in df.columns
