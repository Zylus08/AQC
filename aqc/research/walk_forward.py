"""
aqc/research/walk_forward.py
=============================
Walk-Forward Optimisation (WFO) Engine.

Walk-forward testing is the gold standard for preventing in-sample
overfitting.  The procedure is:

1. Split the full date range into *folds*.
2. For each fold:
   a. **Train window** — optimise strategy parameters using an in-sample
      backtest.
   b. **Test window** — evaluate the *best* parameters found during
      training on unseen out-of-sample data.
3. Aggregate test-window results across all folds to obtain a realistic
   estimate of live performance.

Three window modes are supported:

Rolling (sliding train window)
    Each fold's train window is a fixed-length period that shifts forward.
    Prevents older data from dominating the parameter search.

    Fold 0: Train [T0, T1]  Test [T1, T2]
    Fold 1: Train [T1, T2]  Test [T2, T3]

Expanding (anchored train window)
    The train window always starts at the beginning of the dataset and
    expands fold-by-fold.  Good for strategies that benefit from long
    lookback periods.

    Fold 0: Train [T0, T1]  Test [T1, T2]
    Fold 1: Train [T0, T2]  Test [T2, T3]

Anchored (alias for Expanding)
    Same as expanding.

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

from aqc.research.optimizer import (
    GridSearchOptimizer,
    ObjectiveMetric,
    OptimizationResult,
    RandomSearchOptimizer,
)
from aqc.research.parameter_space import ParameterSpace

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Window mode enum
# ---------------------------------------------------------------------------


class WalkForwardMode(Enum):
    """Walk-forward window mode.

    Attributes
    ----------
    ROLLING:
        Sliding fixed-length train window.
    EXPANDING:
        Expanding train window anchored at the dataset start.
    ANCHORED:
        Alias for :attr:`EXPANDING`.
    """

    ROLLING = auto()
    EXPANDING = auto()
    ANCHORED = auto()  # alias for expanding


# ---------------------------------------------------------------------------
# Fold dataclass
# ---------------------------------------------------------------------------


@dataclass
class WalkForwardFold:
    """Metadata and results for a single walk-forward fold.

    Attributes
    ----------
    fold_index:
        Zero-based fold number.
    train_start:
        Start of the training window (inclusive).
    train_end:
        End of the training window (inclusive).
    test_start:
        Start of the out-of-sample test window (inclusive).
    test_end:
        End of the out-of-sample test window (inclusive).
    best_params:
        Best parameters found during optimisation of the train window.
    train_metrics:
        Performance metrics of the best parameters *on the training data*.
    test_metrics:
        Performance metrics of the best parameters *on the test data*.
    train_equity_curve:
        Equity curve from the training window backtest.
    test_equity_curve:
        Equity curve from the test window backtest.
    optimization_result:
        Full :class:`~aqc.research.optimizer.OptimizationResult` from the
        training window optimisation.
    elapsed_seconds:
        Total wall-clock time for this fold (train optimise + test run).
    """

    fold_index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    best_params: dict[str, Any] = field(default_factory=dict)
    train_metrics: dict[str, Any] = field(default_factory=dict)
    test_metrics: dict[str, Any] = field(default_factory=dict)
    train_equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    test_equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    optimization_result: Optional[OptimizationResult] = None
    elapsed_seconds: float = 0.0

    @property
    def train_sharpe(self) -> float:
        """In-sample Sharpe Ratio for this fold."""
        return float(self.train_metrics.get("sharpe_ratio", float("nan")))

    @property
    def test_sharpe(self) -> float:
        """Out-of-sample Sharpe Ratio for this fold."""
        return float(self.test_metrics.get("sharpe_ratio", float("nan")))

    @property
    def train_return(self) -> float:
        """In-sample total return percentage."""
        return float(self.train_metrics.get("total_return_pct", float("nan")))

    @property
    def test_return(self) -> float:
        """Out-of-sample total return percentage."""
        return float(self.test_metrics.get("total_return_pct", float("nan")))

    def to_dict(self) -> dict[str, Any]:
        """Flatten the fold into a single dict for CSV export.

        Returns
        -------
        dict
        """
        row: dict[str, Any] = {
            "fold": self.fold_index,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
        }
        # Best params
        for k, v in self.best_params.items():
            row[f"param_{k}"] = v
        # Train metrics (prefixed)
        for k, v in self.train_metrics.items():
            row[f"train_{k}"] = v
        # Test metrics (prefixed)
        for k, v in self.test_metrics.items():
            row[f"test_{k}"] = v
        return row

    def __repr__(self) -> str:
        return (
            f"WalkForwardFold(fold={self.fold_index}, "
            f"train=[{self.train_start.date()}, {self.train_end.date()}], "
            f"test=[{self.test_start.date()}, {self.test_end.date()}], "
            f"test_sharpe={self.test_sharpe:.3f})"
        )


# ---------------------------------------------------------------------------
# Walk-forward result
# ---------------------------------------------------------------------------


@dataclass
class WalkForwardResult:
    """Aggregated results across all walk-forward folds.

    Attributes
    ----------
    folds:
        List of :class:`WalkForwardFold` objects, one per time period.
    mode:
        The :class:`WalkForwardMode` used.
    objective_metric:
        The objective that was maximised during each fold's optimisation.
    total_elapsed_seconds:
        Wall-clock time for the entire WFO run.
    """

    folds: list[WalkForwardFold]
    mode: WalkForwardMode
    objective_metric: ObjectiveMetric
    total_elapsed_seconds: float

    # ------------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------------

    def aggregate_metrics(self) -> dict[str, Any]:
        """Compute aggregate statistics across all test folds.

        Returns
        -------
        dict
            Mean, std, min, max for key metrics across test windows.
        """
        import math
        import numpy as np

        metrics_of_interest = [
            "sharpe_ratio", "sortino_ratio", "cagr",
            "max_drawdown_pct", "win_rate", "profit_factor",
            "total_return_pct", "calmar_ratio",
        ]
        agg: dict[str, Any] = {
            "n_folds": len(self.folds),
            "mode": self.mode.name,
            "objective": self.objective_metric.value,
            "total_elapsed_seconds": round(self.total_elapsed_seconds, 3),
        }

        for metric in metrics_of_interest:
            values = [
                f.test_metrics.get(metric)
                for f in self.folds
                if f.test_metrics.get(metric) is not None
            ]
            finite_values = [
                v for v in values
                if isinstance(v, (int, float)) and math.isfinite(v)
            ]
            if finite_values:
                arr = np.array(finite_values, dtype=float)
                agg[f"test_{metric}_mean"] = round(float(arr.mean()), 4)
                agg[f"test_{metric}_std"] = round(float(arr.std()), 4)
                agg[f"test_{metric}_min"] = round(float(arr.min()), 4)
                agg[f"test_{metric}_max"] = round(float(arr.max()), 4)
            else:
                agg[f"test_{metric}_mean"] = float("nan")
                agg[f"test_{metric}_std"] = float("nan")
                agg[f"test_{metric}_min"] = float("nan")
                agg[f"test_{metric}_max"] = float("nan")

        return agg

    def parameter_stability(self) -> dict[str, Any]:
        """Analyse stability of best parameters across folds.

        Returns
        -------
        dict
            For each parameter: mean, std, and mode across folds.
        """
        import numpy as np
        from collections import Counter

        if not self.folds:
            return {}

        all_params = [f.best_params for f in self.folds]
        param_names = list(all_params[0].keys()) if all_params else []

        stability: dict[str, Any] = {}
        for name in param_names:
            values = [p[name] for p in all_params if name in p]
            numeric_values = [v for v in values if isinstance(v, (int, float))]
            if numeric_values:
                arr = np.array(numeric_values, dtype=float)
                stability[name] = {
                    "mean": round(float(arr.mean()), 4),
                    "std": round(float(arr.std()), 4),
                    "min": float(arr.min()),
                    "max": float(arr.max()),
                    "cv": round(float(arr.std() / arr.mean()), 4) if arr.mean() != 0 else float("nan"),
                    "values_per_fold": values,
                }
            else:
                counter = Counter(values)
                stability[name] = {
                    "mode": counter.most_common(1)[0][0] if counter else None,
                    "counts": dict(counter),
                    "values_per_fold": values,
                }
        return stability

    def combined_test_equity_curve(self) -> pd.DataFrame:
        """Concatenate test equity curves from all folds in order.

        Returns a single continuous equity curve spanning the entire
        out-of-sample period.  Each fold's equity is scaled to continue
        from where the previous fold ended to produce a smooth curve.

        Returns
        -------
        pd.DataFrame
            Combined equity curve with columns: ``equity``, ``fold``.
        """
        curves = []
        running_equity = None

        for fold in self.folds:
            curve = fold.test_equity_curve.copy()
            if curve.empty:
                continue

            curve["fold"] = fold.fold_index

            if running_equity is not None:
                # Scale so fold starts where previous ended
                scale = running_equity / curve["equity"].iloc[0]
                curve["equity"] = curve["equity"] * scale

            running_equity = float(curve["equity"].iloc[-1])
            curves.append(curve)

        if not curves:
            return pd.DataFrame()

        combined = pd.concat(curves, axis=0)
        combined = combined[~combined.index.duplicated(keep="last")]
        return combined.sort_index()

    def to_dataframe(self) -> pd.DataFrame:
        """Export per-fold results as a :class:`~pandas.DataFrame`.

        Returns
        -------
        pd.DataFrame
            One row per fold.
        """
        return pd.DataFrame([f.to_dict() for f in self.folds])

    def save_csv(self, filepath: str) -> None:
        """Save fold results to a CSV file.

        Parameters
        ----------
        filepath:
            Output path.  Parent directories are created if missing.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = self.to_dataframe()
        df.to_csv(path, index=False)
        logger.info("Walk-forward results saved to %s (%d rows)", path, len(df))

    def __repr__(self) -> str:
        agg = self.aggregate_metrics()
        mean_sharpe = agg.get("test_sharpe_ratio_mean", float("nan"))
        return (
            f"WalkForwardResult("
            f"folds={len(self.folds)}, "
            f"mode={self.mode.name}, "
            f"mean_test_sharpe={mean_sharpe:.3f})"
        )


# ---------------------------------------------------------------------------
# Strategy factory type
# ---------------------------------------------------------------------------

# Callable that takes (event_queue, symbols, **params) and returns a strategy.
StrategyFactory = Callable[..., Any]

# Callable that takes (data_slice, strategy, params) and returns metrics dict.
BacktestRunner = Callable[[dict[str, pd.DataFrame], Any, dict[str, Any]], dict]


# ---------------------------------------------------------------------------
# Walk-Forward Engine
# ---------------------------------------------------------------------------


class WalkForwardEngine:
    """Orchestrates a full walk-forward optimisation study.

    The engine:

    1. Generates train/test fold windows from the full date range.
    2. For each fold, runs a parameter optimisation on the training window.
    3. Runs a single backtest on the test window using the best parameters.
    4. Aggregates fold results into a :class:`WalkForwardResult`.

    Parameters
    ----------
    data:
        ``{symbol: OHLCV DataFrame}`` mapping.  The DataFrames must span
        the full date range (both train and test periods).
    strategy_factory:
        A callable that accepts ``(event_queue, symbols, **params)`` and
        returns an instantiated :class:`~aqc.strategies.base_strategy.BaseStrategy`.
    backtest_runner:
        A callable that accepts ``(data_slice, strategy, params)`` and
        returns the results dict from
        :meth:`~aqc.backtester.engine.BacktestEngine.run`.  A default
        runner is provided via :meth:`build_default_runner`.
    parameter_space:
        The :class:`~aqc.research.parameter_space.ParameterSpace` to search.
    mode:
        Window mode — :attr:`~WalkForwardMode.ROLLING` (default) or
        :attr:`~WalkForwardMode.EXPANDING`.
    train_period:
        Length of each training window as a pandas offset string or integer
        number of bars.  Examples: ``"365D"`` (365 days), ``252`` (252 bars).
    test_period:
        Length of each test window.  Same format as *train_period*.
    n_folds:
        Number of folds to run.  If ``None``, generates as many non-overlapping
        folds as fit in the data.
    optimizer:
        Search algorithm — ``"grid"`` (default) or ``"random"``.
    n_iter:
        For random search only: number of random samples per fold.
    objective:
        Metric to maximise during training optimisation.
    initial_capital:
        Starting capital for each backtest (default 100 000).
    default_quantity:
        Default order quantity for the portfolio (default 100).
    seed:
        Random seed for reproducibility.

    Examples
    --------
    >>> from aqc.research import WalkForwardEngine, WalkForwardMode
    >>> from aqc.research import ParameterSpace, IntParam
    >>> from aqc.research import ObjectiveMetric
    >>>
    >>> space = ParameterSpace()
    >>> space.add(IntParam("fast_period", 5, 30, step=5))
    >>> space.add(IntParam("slow_period", 20, 80, step=10))
    >>>
    >>> engine = WalkForwardEngine(
    ...     data=data,
    ...     strategy_factory=SMACrossoverStrategy,
    ...     parameter_space=space,
    ...     mode=WalkForwardMode.ROLLING,
    ...     train_period="365D",
    ...     test_period="90D",
    ...     optimizer="grid",
    ...     objective=ObjectiveMetric.SHARPE,
    ... )
    >>> result = engine.run()
    >>> result.save_csv("reports/walk_forward_results.csv")
    """

    def __init__(
        self,
        data: dict[str, pd.DataFrame],
        strategy_factory: StrategyFactory,
        parameter_space: ParameterSpace,
        backtest_runner: Optional[BacktestRunner] = None,
        mode: WalkForwardMode = WalkForwardMode.ROLLING,
        train_period: str | int = "365D",
        test_period: str | int = "90D",
        n_folds: Optional[int] = None,
        optimizer: str = "grid",
        n_iter: int = 50,
        objective: ObjectiveMetric = ObjectiveMetric.SHARPE,
        initial_capital: float = 100_000.0,
        default_quantity: float = 100.0,
        seed: int = 42,
    ) -> None:
        self.data = data
        self.strategy_factory = strategy_factory
        self.parameter_space = parameter_space
        self.mode = mode
        self.train_period = train_period
        self.test_period = test_period
        self.n_folds = n_folds
        self.optimizer_type = optimizer
        self.n_iter = n_iter
        self.objective = objective
        self.initial_capital = initial_capital
        self.default_quantity = default_quantity
        self.seed = seed

        # Use provided runner or build the default
        self._backtest_runner = backtest_runner or self._build_default_runner()

        # Compute the common index across all symbols
        self._common_index = self._get_common_index()
        self._symbols = list(data.keys())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> WalkForwardResult:
        """Execute the full walk-forward optimisation.

        Returns
        -------
        WalkForwardResult
        """
        folds_meta = self._generate_folds()

        logger.info("=" * 65)
        logger.info("AQC Walk-Forward Optimisation")
        logger.info("  Mode      : %s", self.mode.name)
        logger.info("  Optimizer : %s", self.optimizer_type.upper())
        logger.info("  Objective : %s", self.objective.value)
        logger.info("  Folds     : %d", len(folds_meta))
        logger.info("  Symbols   : %s", self._symbols)
        logger.info("  Grid size : %d", self.parameter_space.grid_size())
        logger.info("=" * 65)

        t_start = time.perf_counter()
        completed_folds: list[WalkForwardFold] = []

        for i, (train_start, train_end, test_start, test_end) in enumerate(folds_meta):
            logger.info(
                "Fold %d/%d  train=[%s, %s]  test=[%s, %s]",
                i + 1,
                len(folds_meta),
                train_start.date(),
                train_end.date(),
                test_start.date(),
                test_end.date(),
            )
            fold = self._run_fold(i, train_start, train_end, test_start, test_end)
            completed_folds.append(fold)

            logger.info(
                "  Fold %d complete: best_params=%s  "
                "train_%s=%.4f  test_%s=%.4f  (%.1fs)",
                i + 1,
                fold.best_params,
                self.objective.value,
                fold.train_metrics.get(self.objective.value, float("nan")),
                self.objective.value,
                fold.test_metrics.get(self.objective.value, float("nan")),
                fold.elapsed_seconds,
            )

        total_elapsed = time.perf_counter() - t_start
        result = WalkForwardResult(
            folds=completed_folds,
            mode=self.mode,
            objective_metric=self.objective,
            total_elapsed_seconds=total_elapsed,
        )

        agg = result.aggregate_metrics()
        logger.info("=" * 65)
        logger.info("Walk-Forward Complete  (%.1f s total)", total_elapsed)
        logger.info(
            "  Mean test Sharpe  : %.4f +/- %.4f",
            agg.get("test_sharpe_ratio_mean", float("nan")),
            agg.get("test_sharpe_ratio_std", float("nan")),
        )
        logger.info(
            "  Mean test Return  : %.2f%%",
            agg.get("test_total_return_pct_mean", float("nan")),
        )
        logger.info(
            "  Mean test MaxDD   : %.2f%%",
            agg.get("test_max_drawdown_pct_mean", float("nan")),
        )
        logger.info("=" * 65)

        return result

    # ------------------------------------------------------------------
    # Fold runner
    # ------------------------------------------------------------------

    def _run_fold(
        self,
        fold_index: int,
        train_start: pd.Timestamp,
        train_end: pd.Timestamp,
        test_start: pd.Timestamp,
        test_end: pd.Timestamp,
    ) -> WalkForwardFold:
        """Optimise on the training window, then evaluate on the test window.

        Parameters
        ----------
        fold_index:
            Zero-based fold number.
        train_start / train_end:
            Training window boundaries.
        test_start / test_end:
            Test window boundaries.

        Returns
        -------
        WalkForwardFold
        """
        t0 = time.perf_counter()

        # Slice data for train and test
        train_data = self._slice_data(train_start, train_end)
        test_data = self._slice_data(test_start, test_end)

        # ── Step 1: Optimise on training window ──────────────────────
        def train_backtest_fn(params: dict[str, Any]) -> dict:
            return self._backtest_runner(train_data, self.strategy_factory, params)

        opt_result = self._build_optimizer(train_backtest_fn).run()

        best_params = opt_result.best_params
        train_metrics = opt_result.best_metrics

        # Retrieve the train equity curve from the best evaluation
        # by running one more backtest with the best parameters
        train_full = self._backtest_runner(train_data, self.strategy_factory, best_params)
        train_equity = train_full.get("equity_curve", pd.DataFrame())

        # ── Step 2: Evaluate on test window ──────────────────────────
        test_result = self._backtest_runner(test_data, self.strategy_factory, best_params)
        test_metrics = test_result.get("performance_metrics", {})
        test_equity = test_result.get("equity_curve", pd.DataFrame())

        elapsed = time.perf_counter() - t0

        return WalkForwardFold(
            fold_index=fold_index,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            best_params=best_params,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            train_equity_curve=train_equity,
            test_equity_curve=test_equity,
            optimization_result=opt_result,
            elapsed_seconds=elapsed,
        )

    # ------------------------------------------------------------------
    # Fold generation
    # ------------------------------------------------------------------

    def _generate_folds(
        self,
    ) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """Generate (train_start, train_end, test_start, test_end) tuples.

        Returns
        -------
        list of 4-tuples
        """
        idx = self._common_index
        if len(idx) == 0:
            raise ValueError("Dataset is empty — cannot generate folds.")

        train_bars = self._period_to_bars(self.train_period, idx)
        test_bars = self._period_to_bars(self.test_period, idx)

        if train_bars + test_bars > len(idx):
            raise ValueError(
                f"train_period ({train_bars} bars) + test_period ({test_bars} bars) "
                f"exceeds available data ({len(idx)} bars)."
            )

        folds = []
        fold_num = 0

        if self.mode == WalkForwardMode.ROLLING:
            # Sliding window: train window shifts forward by test_bars each fold
            pos = 0
            while pos + train_bars + test_bars <= len(idx):
                train_start = idx[pos]
                train_end = idx[pos + train_bars - 1]
                test_start = idx[pos + train_bars]
                test_end = idx[min(pos + train_bars + test_bars - 1, len(idx) - 1)]
                folds.append((train_start, train_end, test_start, test_end))
                pos += test_bars
                fold_num += 1
                if self.n_folds and fold_num >= self.n_folds:
                    break

        elif self.mode in (WalkForwardMode.EXPANDING, WalkForwardMode.ANCHORED):
            # Expanding window: train always starts at idx[0]
            pos = train_bars
            while pos + test_bars <= len(idx):
                train_start = idx[0]
                train_end = idx[pos - 1]
                test_start = idx[pos]
                test_end = idx[min(pos + test_bars - 1, len(idx) - 1)]
                folds.append((train_start, train_end, test_start, test_end))
                pos += test_bars
                fold_num += 1
                if self.n_folds and fold_num >= self.n_folds:
                    break

        if not folds:
            raise ValueError(
                "No folds could be generated.  Check that train_period and "
                "test_period together are shorter than the dataset."
            )

        return folds

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _slice_data(
        self, start: pd.Timestamp, end: pd.Timestamp
    ) -> dict[str, pd.DataFrame]:
        """Slice all symbol DataFrames to the [start, end] range.

        Parameters
        ----------
        start / end:
            Inclusive timestamp boundaries.

        Returns
        -------
        dict[str, pd.DataFrame]
        """
        return {
            symbol: df.loc[start:end].copy()
            for symbol, df in self.data.items()
        }

    def _get_common_index(self) -> pd.DatetimeIndex:
        """Return the sorted union of all symbols' DatetimeIndexes."""
        indices = [df.index for df in self.data.values()]
        if not indices:
            return pd.DatetimeIndex([])
        combined = indices[0]
        for idx in indices[1:]:
            combined = combined.union(idx)
        return combined.sort_values()

    @staticmethod
    def _period_to_bars(period: str | int, index: pd.DatetimeIndex) -> int:
        """Convert a period specification to a bar count.

        Parameters
        ----------
        period:
            Either an integer (number of bars) or a pandas offset string
            (e.g. ``"365D"``).
        index:
            The DatetimeIndex of the full dataset.

        Returns
        -------
        int
            Number of bars corresponding to the period.
        """
        if isinstance(period, int):
            return period

        # Parse offset string — count bars in the first period-length window
        start = index[0]
        end = start + pd.tseries.frequencies.to_offset(period)
        n_bars = int((index <= end).sum())
        return max(n_bars, 2)

    def _build_optimizer(self, backtest_fn: Callable) -> GridSearchOptimizer | RandomSearchOptimizer:
        """Instantiate the configured optimizer.

        Parameters
        ----------
        backtest_fn:
            Backtest callable for the current fold.

        Returns
        -------
        GridSearchOptimizer | RandomSearchOptimizer
        """
        if self.optimizer_type == "random":
            return RandomSearchOptimizer(
                backtest_fn=backtest_fn,
                space=self.parameter_space,
                n_iter=self.n_iter,
                objective=self.objective,
                seed=self.seed,
                verbose=False,
            )
        else:
            return GridSearchOptimizer(
                backtest_fn=backtest_fn,
                space=self.parameter_space,
                objective=self.objective,
                verbose=False,
            )

    # ------------------------------------------------------------------
    # Default backtest runner
    # ------------------------------------------------------------------

    def _build_default_runner(self) -> BacktestRunner:
        """Build a default backtest runner using the AQC engine stack.

        Returns a callable that:
        1. Assembles the full AQC engine stack (queue, portfolio, broker, etc.)
        2. Runs the strategy on the given data slice.
        3. Returns the results dict.

        Returns
        -------
        BacktestRunner
        """
        initial_capital = self.initial_capital
        default_quantity = self.default_quantity

        def runner(
            data_slice: dict[str, pd.DataFrame],
            strategy_factory: StrategyFactory,
            params: dict[str, Any],
        ) -> dict:
            # Local imports to avoid circular imports at module level
            from aqc.backtester.event_queue import EventQueue
            from aqc.backtester.broker import (
                SimulatedBroker,
                PercentageCommission,
                FixedBpsSlippage,
            )
            from aqc.backtester.execution import ExecutionEngine
            from aqc.backtester.portfolio import Portfolio
            from aqc.backtester.engine import BacktestEngine
            from aqc.risk.risk_manager import RiskManager, RiskConfig

            symbols = list(data_slice.keys())

            eq = EventQueue()
            risk = RiskManager(
                config=RiskConfig(
                    max_position_pct_equity=1.0,
                    max_gross_exposure_pct=5.0,
                    max_daily_loss_pct=0.99,
                    max_open_positions=20,
                )
            )
            portfolio = Portfolio(
                event_queue=eq,
                risk_manager=risk,
                initial_capital=initial_capital,
                default_quantity=default_quantity,
            )
            risk.reset_daily_state(initial_capital)

            broker = SimulatedBroker(
                event_queue=eq,
                commission_model=PercentageCommission(rate=0.001),
                slippage_model=FixedBpsSlippage(bps=5),
            )
            exec_engine = ExecutionEngine(broker=broker, event_queue=eq)

            # Instantiate the strategy with the current parameter set
            strategy = strategy_factory(
                event_queue=eq,
                symbols=symbols,
                **params,
            )

            # Suppress verbose engine logging during optimisation
            engine_logger = logging.getLogger("aqc.backtester.engine")
            original_level = engine_logger.level
            engine_logger.setLevel(logging.WARNING)

            engine = BacktestEngine(
                data=data_slice,
                strategy=strategy,
                portfolio=portfolio,
                execution_engine=exec_engine,
                event_queue=eq,
            )

            try:
                # Suppress the report printout during optimisation
                import aqc.analytics.reporting as rep_module
                original_print = rep_module.ReportGenerator.print_report
                rep_module.ReportGenerator.print_report = lambda self: None

                results = engine.run()

                rep_module.ReportGenerator.print_report = original_print
            finally:
                engine_logger.setLevel(original_level)

            return results

        return runner

    def __repr__(self) -> str:
        return (
            f"WalkForwardEngine("
            f"mode={self.mode.name}, "
            f"optimizer={self.optimizer_type}, "
            f"objective={self.objective.value}, "
            f"symbols={self._symbols})"
        )
