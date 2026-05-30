"""
aqc/research/comparison/comparator.py
=======================================
Comparative Backtesting Engine.

Runs multiple backtest configurations on the same dataset and produces
side-by-side performance comparisons with statistical significance tests.

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from aqc.analytics.metrics import PerformanceMetrics

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Container for a single backtest variant's results.

    Attributes
    ----------
    name:
        Human-readable label (e.g. "Baseline", "Vol-Targeted").
    equity_curve:
        Equity curve DataFrame.
    trade_log:
        Trade log list.
    metrics:
        Computed performance metrics dict.
    extra:
        Any additional metadata.
    """

    name: str = ""
    equity_curve: Optional[pd.DataFrame] = None
    trade_log: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)


class BacktestComparator:
    """Runs and compares multiple backtest configurations.

    This class stores results from different backtest runs and provides
    side-by-side metric comparison and statistical tests.

    Examples
    --------
    >>> comparator = BacktestComparator()
    >>> comparator.add_result("Baseline", eq_curve1, trades1)
    >>> comparator.add_result("Vol-Target", eq_curve2, trades2)
    >>> comparison = comparator.compare()
    """

    def __init__(self) -> None:
        self.results: dict[str, BacktestResult] = {}

    def add_result(
        self,
        name: str,
        equity_curve: pd.DataFrame,
        trade_log: list,
        extra: Optional[dict] = None,
    ) -> None:
        """Add a backtest result for comparison.

        Parameters
        ----------
        name:
            Variant label.
        equity_curve:
            Equity curve DataFrame with ``equity`` column.
        trade_log:
            Trade log list.
        extra:
            Additional metadata to attach.
        """
        metrics_obj = PerformanceMetrics(equity_curve, trade_log)
        metrics = metrics_obj.compute_all()

        self.results[name] = BacktestResult(
            name=name,
            equity_curve=equity_curve,
            trade_log=trade_log,
            metrics=metrics,
            extra=extra or {},
        )
        logger.info("Added result '%s': %d bars, %d trades", name, len(equity_curve), len(trade_log))

    def compare(self) -> pd.DataFrame:
        """Produce a side-by-side metric comparison.

        Returns
        -------
        pd.DataFrame
            Rows = metric names, columns = variant names.
        """
        if not self.results:
            return pd.DataFrame()

        data = {}
        for name, result in self.results.items():
            data[name] = result.metrics

        df = pd.DataFrame(data)
        return df

    def get_returns(self, name: str) -> pd.Series:
        """Get daily returns for a named result.

        Parameters
        ----------
        name:
            Variant label.

        Returns
        -------
        pd.Series
        """
        result = self.results.get(name)
        if result is None or result.equity_curve is None or result.equity_curve.empty:
            return pd.Series(dtype=float)
        return result.equity_curve["equity"].pct_change().dropna()

    def get_all_equity_curves(self) -> pd.DataFrame:
        """Combine all equity curves into a single DataFrame.

        Returns
        -------
        pd.DataFrame
            Columns = variant names, values = equity.
        """
        curves = {}
        for name, result in self.results.items():
            if result.equity_curve is not None and not result.equity_curve.empty:
                curves[name] = result.equity_curve["equity"]
        return pd.DataFrame(curves)


class StatisticalTests:
    """Statistical significance tests for backtest comparison.

    Examples
    --------
    >>> tests = StatisticalTests()
    >>> result = tests.sharpe_difference_test(returns_a, returns_b)
    """

    @staticmethod
    def t_test_returns(
        returns_a: pd.Series,
        returns_b: pd.Series,
    ) -> dict:
        """Two-sample t-test on daily returns.

        Tests H0: mean(A) == mean(B).

        Parameters
        ----------
        returns_a:
            Returns from variant A.
        returns_b:
            Returns from variant B.

        Returns
        -------
        dict
            t-statistic, p-value, significant at 5%.
        """
        from scipy import stats

        a = returns_a.dropna().values
        b = returns_b.dropna().values

        if len(a) < 2 or len(b) < 2:
            return {"t_stat": np.nan, "p_value": np.nan, "significant_5pct": False}

        t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)

        return {
            "t_stat": round(float(t_stat), 4),
            "p_value": round(float(p_value), 6),
            "significant_5pct": p_value < 0.05,
        }

    @staticmethod
    def bootstrap_sharpe_ci(
        returns: pd.Series,
        n_bootstrap: int = 1000,
        ci_level: float = 0.95,
        ann_factor: int = 252,
        seed: int = 42,
    ) -> dict:
        """Bootstrap confidence intervals for Sharpe ratio.

        Parameters
        ----------
        returns:
            Daily returns.
        n_bootstrap:
            Number of bootstrap samples.
        ci_level:
            Confidence level (default 0.95).
        ann_factor:
            Annualisation factor.
        seed:
            Random seed.

        Returns
        -------
        dict
            point_estimate, ci_lower, ci_upper.
        """
        r = returns.dropna().values
        if len(r) < 10:
            return {"point_estimate": np.nan, "ci_lower": np.nan, "ci_upper": np.nan}

        rng = np.random.default_rng(seed)
        sharpes = []

        for _ in range(n_bootstrap):
            sample = rng.choice(r, size=len(r), replace=True)
            if np.std(sample) < 1e-10:
                continue
            sr = float(np.mean(sample) / np.std(sample) * math.sqrt(ann_factor))
            sharpes.append(sr)

        if not sharpes:
            return {"point_estimate": np.nan, "ci_lower": np.nan, "ci_upper": np.nan}

        alpha = (1.0 - ci_level) / 2
        ci_lower = float(np.percentile(sharpes, alpha * 100))
        ci_upper = float(np.percentile(sharpes, (1.0 - alpha) * 100))
        point = float(np.mean(sharpes))

        return {
            "point_estimate": round(point, 4),
            "ci_lower": round(ci_lower, 4),
            "ci_upper": round(ci_upper, 4),
        }

    @staticmethod
    def sharpe_difference_test(
        returns_a: pd.Series,
        returns_b: pd.Series,
        n_bootstrap: int = 1000,
        ann_factor: int = 252,
        seed: int = 42,
    ) -> dict:
        """Test whether Sharpe(A) > Sharpe(B) with bootstrap.

        Parameters
        ----------
        returns_a:
            Returns from enhanced strategy.
        returns_b:
            Returns from baseline.
        n_bootstrap:
            Bootstrap samples.
        ann_factor:
            Annualisation factor.
        seed:
            Random seed.

        Returns
        -------
        dict
            sharpe_diff, p_value (one-sided), significant at 5%.
        """
        a = returns_a.dropna().values
        b = returns_b.dropna().values
        n = min(len(a), len(b))

        if n < 10:
            return {"sharpe_diff": np.nan, "p_value": np.nan, "significant_5pct": False}

        a = a[:n]
        b = b[:n]

        rng = np.random.default_rng(seed)
        diffs = []

        for _ in range(n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            sa = a[idx]
            sb = b[idx]
            std_a = np.std(sa)
            std_b = np.std(sb)
            if std_a < 1e-10 or std_b < 1e-10:
                continue
            sr_a = np.mean(sa) / std_a * math.sqrt(ann_factor)
            sr_b = np.mean(sb) / std_b * math.sqrt(ann_factor)
            diffs.append(sr_a - sr_b)

        if not diffs:
            return {"sharpe_diff": np.nan, "p_value": np.nan, "significant_5pct": False}

        # Point estimate
        std_a = np.std(a)
        std_b = np.std(b)
        if std_a < 1e-10 or std_b < 1e-10:
            point_diff = 0.0
        else:
            point_diff = float(
                (np.mean(a) / std_a - np.mean(b) / std_b) * math.sqrt(ann_factor)
            )

        # P-value: fraction of bootstrap where A ≤ B
        p_value = float(np.mean(np.array(diffs) <= 0))

        return {
            "sharpe_diff": round(point_diff, 4),
            "p_value": round(p_value, 4),
            "significant_5pct": p_value < 0.05,
        }
