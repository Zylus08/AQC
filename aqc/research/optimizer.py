"""
aqc/research/optimizer.py
==========================
Strategy parameter optimizers — Grid Search and Random Search.

The :class:`ParameterOptimizer` protocol defines a common interface so that
calling code never depends on a specific search algorithm.  New optimizers
(Bayesian, evolutionary, etc.) can be added by implementing the protocol.

Architecture
------------
Each optimizer:

1. Receives a callable ``backtest_fn`` that accepts a ``dict[str, Any]``
   of parameters and returns a ``dict`` of performance metrics (the output
   of :meth:`~aqc.backtester.engine.BacktestEngine.run`).

2. Evaluates the **objective metric** (e.g. Sharpe Ratio) for each candidate.

3. Returns an :class:`OptimizationResult` containing the best parameters
   and the full evaluation history.

Objective Metrics
-----------------
The ``ObjectiveMetric`` enum maps user-friendly labels to the keys used in
the ``performance_metrics`` sub-dict returned by the backtest engine.

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from aqc.research.parameter_space import ParameterGrid, ParameterSpace

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Objective metric enum
# ---------------------------------------------------------------------------


class ObjectiveMetric(Enum):
    """Performance metrics available as optimisation objectives.

    The string value must match the key in the ``performance_metrics`` dict
    returned by :meth:`~aqc.backtester.engine.BacktestEngine.run`.
    """

    SHARPE = "sharpe_ratio"
    SORTINO = "sortino_ratio"
    CALMAR = "calmar_ratio"
    CAGR = "cagr"
    MAX_DRAWDOWN = "max_drawdown_pct"   # minimise (most negative = worst)
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    TOTAL_RETURN = "total_return_pct"


# ---------------------------------------------------------------------------
# Evaluation record
# ---------------------------------------------------------------------------


@dataclass
class EvaluationRecord:
    """A single parameter evaluation during optimisation.

    Attributes
    ----------
    params:
        The parameter combination evaluated.
    metrics:
        Full metrics dictionary returned by the backtest.
    objective_value:
        The scalar value of the chosen objective metric.
    elapsed_seconds:
        Wall-clock time taken for this evaluation.
    """

    params: dict[str, Any]
    metrics: dict[str, Any]
    objective_value: float
    elapsed_seconds: float


# ---------------------------------------------------------------------------
# Optimisation result
# ---------------------------------------------------------------------------


@dataclass
class OptimizationResult:
    """Aggregated result of a full parameter optimisation run.

    Attributes
    ----------
    best_params:
        Parameter combination that maximised the objective metric.
    best_value:
        Objective metric value for the best parameters.
    best_metrics:
        Full metrics dict for the best parameter set.
    all_evaluations:
        All :class:`EvaluationRecord` objects, ordered by evaluation.
    objective_metric:
        The :class:`ObjectiveMetric` that was optimised.
    total_evaluations:
        Total number of parameter combinations evaluated.
    total_elapsed_seconds:
        Total wall-clock time for the entire optimisation.
    """

    best_params: dict[str, Any]
    best_value: float
    best_metrics: dict[str, Any]
    all_evaluations: list[EvaluationRecord]
    objective_metric: ObjectiveMetric
    total_evaluations: int
    total_elapsed_seconds: float

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def top_n(self, n: int = 5) -> list[EvaluationRecord]:
        """Return the top *n* evaluations sorted by objective value descending.

        For ``MAX_DRAWDOWN`` the metric is negated so higher = better is
        preserved across all objectives.

        Parameters
        ----------
        n:
            Number of top results to return.

        Returns
        -------
        list[EvaluationRecord]
        """
        sorted_evals = sorted(
            self.all_evaluations,
            key=lambda e: e.objective_value,
            reverse=True,
        )
        return sorted_evals[:n]

    def to_dataframe(self):
        """Export all evaluations as a :class:`~pandas.DataFrame`.

        Returns
        -------
        pd.DataFrame
            One row per evaluation with parameter columns and metric columns.
        """
        import pandas as pd

        rows = []
        for ev in self.all_evaluations:
            row = dict(ev.params)
            row["objective_value"] = ev.objective_value
            row["elapsed_seconds"] = ev.elapsed_seconds
            row.update({f"metric_{k}": v for k, v in ev.metrics.items()})
            rows.append(row)
        return pd.DataFrame(rows)

    def __repr__(self) -> str:
        return (
            f"OptimizationResult("
            f"best={self.best_params}, "
            f"best_{self.objective_metric.value}={self.best_value:.4f}, "
            f"n_evals={self.total_evaluations})"
        )


# ---------------------------------------------------------------------------
# BacktestFn type alias
# ---------------------------------------------------------------------------

# A callable that accepts a dict of parameters and returns a backtest result dict.
BacktestFn = Callable[[dict[str, Any]], dict]


# ---------------------------------------------------------------------------
# Base optimizer
# ---------------------------------------------------------------------------


class _BaseOptimizer:
    """Internal base class providing shared optimisation infrastructure.

    Parameters
    ----------
    backtest_fn:
        A callable ``(params: dict) -> results: dict``.  The ``results``
        dict must contain a ``"performance_metrics"`` key whose value is
        the standard AQC metrics dictionary.
    space:
        The hyperparameter search space.
    objective:
        Which metric to maximise.  Note: ``MAX_DRAWDOWN`` is automatically
        negated so that the optimizer always *maximises* the objective.
    maximize:
        If ``True`` (default), higher values of the objective are better.
        Set to ``False`` to minimise (handled internally for drawdown).
    """

    def __init__(
        self,
        backtest_fn: BacktestFn,
        space: ParameterSpace,
        objective: ObjectiveMetric = ObjectiveMetric.SHARPE,
        maximize: bool = True,
    ) -> None:
        self._backtest_fn = backtest_fn
        self.space = space
        self.objective = objective
        self.maximize = maximize

        # Drawdown is inherently negative — flip the sign so the optimizer
        # treats higher values as better in all cases.
        self._negate = objective == ObjectiveMetric.MAX_DRAWDOWN

    def _evaluate(self, params: dict[str, Any]) -> EvaluationRecord:
        """Run a single backtest and extract the objective metric value.

        Parameters
        ----------
        params:
            Parameter combination to evaluate.

        Returns
        -------
        EvaluationRecord
        """
        t0 = time.perf_counter()
        result = self._backtest_fn(params)
        elapsed = time.perf_counter() - t0

        perf = result.get("performance_metrics", {})
        raw_value = perf.get(self.objective.value, float("nan"))

        # Guard against NaN / inf
        import math
        if raw_value is None or (isinstance(raw_value, float) and not math.isfinite(raw_value)):
            raw_value = -1e9 if self.maximize else 1e9

        # For drawdown (negative by convention), negate so higher = better
        obj_value = -raw_value if self._negate else raw_value

        return EvaluationRecord(
            params=params,
            metrics=perf,
            objective_value=float(obj_value),
            elapsed_seconds=elapsed,
        )

    def _build_result(
        self,
        evaluations: list[EvaluationRecord],
        total_elapsed: float,
    ) -> OptimizationResult:
        """Compile all evaluations into a final :class:`OptimizationResult`.

        Parameters
        ----------
        evaluations:
            All evaluation records.
        total_elapsed:
            Total elapsed time.

        Returns
        -------
        OptimizationResult
        """
        if not evaluations:
            raise RuntimeError("No evaluations completed — cannot build result.")

        best = max(evaluations, key=lambda e: e.objective_value)
        return OptimizationResult(
            best_params=best.params,
            best_value=best.objective_value,
            best_metrics=best.metrics,
            all_evaluations=evaluations,
            objective_metric=self.objective,
            total_evaluations=len(evaluations),
            total_elapsed_seconds=total_elapsed,
        )


# ---------------------------------------------------------------------------
# Grid Search
# ---------------------------------------------------------------------------


class GridSearchOptimizer(_BaseOptimizer):
    """Exhaustive grid search over all parameter combinations.

    Evaluates every point in the Cartesian product of parameter values.
    Suitable for small search spaces (< ~1 000 combinations).

    Parameters
    ----------
    backtest_fn:
        Callable ``(params: dict) -> results: dict``.
    space:
        Parameter search space.
    objective:
        Metric to maximise (default :attr:`~ObjectiveMetric.SHARPE`).
    verbose:
        If ``True``, log progress after every evaluation.

    Examples
    --------
    >>> optimizer = GridSearchOptimizer(backtest_fn=run_backtest, space=space)
    >>> result = optimizer.run()
    >>> print(result.best_params)
    {'fast_period': 20, 'slow_period': 50}
    """

    def __init__(
        self,
        backtest_fn: BacktestFn,
        space: ParameterSpace,
        objective: ObjectiveMetric = ObjectiveMetric.SHARPE,
        verbose: bool = True,
    ) -> None:
        super().__init__(backtest_fn, space, objective)
        self.verbose = verbose

    def run(self) -> OptimizationResult:
        """Execute the exhaustive grid search.

        Returns
        -------
        OptimizationResult
        """
        grid = ParameterGrid(self.space)
        total = len(grid)

        logger.info(
            "GridSearch starting: %d combinations, objective=%s",
            total,
            self.objective.value,
        )

        evaluations: list[EvaluationRecord] = []
        t_start = time.perf_counter()

        for i, params in enumerate(grid, start=1):
            record = self._evaluate(params)
            evaluations.append(record)

            if self.verbose:
                logger.info(
                    "[%d/%d] %s=%.4f  params=%s  (%.2fs)",
                    i,
                    total,
                    self.objective.value,
                    record.objective_value,
                    params,
                    record.elapsed_seconds,
                )

        total_elapsed = time.perf_counter() - t_start
        result = self._build_result(evaluations, total_elapsed)

        logger.info(
            "GridSearch complete: best %s=%.4f  params=%s  total=%.2fs",
            self.objective.value,
            result.best_value,
            result.best_params,
            total_elapsed,
        )
        return result


# ---------------------------------------------------------------------------
# Random Search
# ---------------------------------------------------------------------------


class RandomSearchOptimizer(_BaseOptimizer):
    """Random search over the parameter space.

    Samples *n_iter* random parameter combinations and evaluates each.
    More efficient than grid search for large continuous spaces.

    Parameters
    ----------
    backtest_fn:
        Callable ``(params: dict) -> results: dict``.
    space:
        Parameter search space.
    n_iter:
        Number of random samples to evaluate.
    objective:
        Metric to maximise (default :attr:`~ObjectiveMetric.SHARPE`).
    seed:
        Random seed for reproducibility.
    verbose:
        If ``True``, log progress after every evaluation.

    Examples
    --------
    >>> optimizer = RandomSearchOptimizer(
    ...     backtest_fn=run_backtest,
    ...     space=space,
    ...     n_iter=50,
    ...     seed=42,
    ... )
    >>> result = optimizer.run()
    """

    def __init__(
        self,
        backtest_fn: BacktestFn,
        space: ParameterSpace,
        n_iter: int = 50,
        objective: ObjectiveMetric = ObjectiveMetric.SHARPE,
        seed: int = 42,
        verbose: bool = True,
    ) -> None:
        super().__init__(backtest_fn, space, objective)
        self.n_iter = n_iter
        self.seed = seed
        self.verbose = verbose

    def run(self) -> OptimizationResult:
        """Execute the random search.

        Returns
        -------
        OptimizationResult
        """
        rng = random.Random(self.seed)

        logger.info(
            "RandomSearch starting: n_iter=%d, objective=%s, seed=%d",
            self.n_iter,
            self.objective.value,
            self.seed,
        )

        evaluations: list[EvaluationRecord] = []
        t_start = time.perf_counter()

        for i in range(1, self.n_iter + 1):
            params = self.space.sample(rng)
            record = self._evaluate(params)
            evaluations.append(record)

            if self.verbose:
                logger.info(
                    "[%d/%d] %s=%.4f  params=%s  (%.2fs)",
                    i,
                    self.n_iter,
                    self.objective.value,
                    record.objective_value,
                    params,
                    record.elapsed_seconds,
                )

        total_elapsed = time.perf_counter() - t_start
        result = self._build_result(evaluations, total_elapsed)

        logger.info(
            "RandomSearch complete: best %s=%.4f  params=%s  total=%.2fs",
            self.objective.value,
            result.best_value,
            result.best_params,
            total_elapsed,
        )
        return result
