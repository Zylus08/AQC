"""
aqc/alpha/alpha_ranker.py
===========================
Multi-criteria alpha ranking engine.

Takes a collection of alphas with their evaluation metrics and produces
a ranked leaderboard with composite scores.  Supports Pareto-optimal
frontier identification for multi-objective selection.

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from aqc.alpha.alpha_base import AlphaBase, AlphaMetrics

logger = logging.getLogger(__name__)


@dataclass
class RankingWeights:
    """Weights for the composite ranking score.

    All weights must be non-negative.  They are normalised internally.

    Attributes
    ----------
    sharpe:
        Weight for Sharpe Ratio.
    ic:
        Weight for Information Coefficient.
    hit_rate:
        Weight for directional accuracy.
    profit_factor:
        Weight for profit factor.
    drawdown_penalty:
        Penalty weight for max drawdown (higher = more penalty).
    turnover_penalty:
        Penalty weight for excessive turnover.
    capacity:
        Weight for capacity estimate.
    decay_penalty:
        Penalty for fast-decaying alphas.
    """

    sharpe: float = 0.30
    ic: float = 0.20
    hit_rate: float = 0.10
    profit_factor: float = 0.10
    drawdown_penalty: float = 0.10
    turnover_penalty: float = 0.05
    capacity: float = 0.10
    decay_penalty: float = 0.05

    def normalised(self) -> dict[str, float]:
        """Return weights normalised to sum to 1."""
        raw = {
            "sharpe": self.sharpe,
            "ic": self.ic,
            "hit_rate": self.hit_rate,
            "profit_factor": self.profit_factor,
            "drawdown_penalty": self.drawdown_penalty,
            "turnover_penalty": self.turnover_penalty,
            "capacity": self.capacity,
            "decay_penalty": self.decay_penalty,
        }
        total = sum(raw.values())
        if total <= 0:
            return {k: 1.0 / len(raw) for k in raw}
        return {k: v / total for k, v in raw.items()}


class AlphaRanker:
    """Rank alphas by a weighted composite score.

    Parameters
    ----------
    weights:
        Ranking weights.  Defaults to a balanced profile.

    Examples
    --------
    >>> ranker = AlphaRanker()
    >>> ranker.add("alpha_a", metrics_a)
    >>> ranker.add("alpha_b", metrics_b)
    >>> leaderboard = ranker.rank()
    """

    def __init__(self, weights: Optional[RankingWeights] = None) -> None:
        self.weights = weights or RankingWeights()
        self._entries: dict[str, AlphaMetrics] = {}

    # ------------------------------------------------------------------
    # Entry management
    # ------------------------------------------------------------------

    def add(self, alpha_name: str, metrics: AlphaMetrics) -> None:
        """Add an alpha's metrics to the ranking pool.

        Parameters
        ----------
        alpha_name:
            Unique identifier.
        metrics:
            Evaluation metrics for the alpha.
        """
        self._entries[alpha_name] = metrics

    def add_from_alpha(self, alpha: AlphaBase) -> None:
        """Add an alpha using its cached metrics.

        Parameters
        ----------
        alpha:
            Alpha instance with ``cached_metrics`` populated.

        Raises
        ------
        ValueError:
            If the alpha has no cached metrics.
        """
        if alpha.cached_metrics is None:
            raise ValueError(
                f"Alpha {alpha.name!r} has no cached metrics.  "
                "Run evaluate() first."
            )
        self.add(alpha.name, alpha.cached_metrics)

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    def rank(self) -> pd.DataFrame:
        """Produce a ranked leaderboard DataFrame.

        Returns
        -------
        pd.DataFrame
            Columns: all metric fields + ``composite_score`` + ``rank``.
            Sorted by ``composite_score`` descending.
        """
        if not self._entries:
            logger.warning("No entries to rank.")
            return pd.DataFrame()

        rows: list[dict] = []
        for name, m in self._entries.items():
            row = m.to_dict()
            row["alpha_name"] = name
            row["composite_score"] = self._composite_score(m)
            rows.append(row)

        df = pd.DataFrame(rows)
        df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)

        # Reorder columns
        cols = ["rank", "alpha_name", "composite_score"] + [
            c for c in df.columns if c not in ("rank", "alpha_name", "composite_score")
        ]
        return df[cols]

    def _composite_score(self, m: AlphaMetrics) -> float:
        """Compute the weighted composite score for a single alpha.

        Parameters
        ----------
        m:
            Alpha metrics.

        Returns
        -------
        float
            Composite score (higher = better).
        """
        w = self.weights.normalised()

        # Positive contributions (higher is better)
        score = (
            w["sharpe"] * self._clip(m.sharpe_ratio, -3, 5)
            + w["ic"] * self._clip(m.information_coefficient, -1, 1)
            + w["hit_rate"] * self._clip(m.hit_rate, 0, 1)
            + w["profit_factor"] * self._clip(m.profit_factor, 0, 10) / 10.0
            + w["capacity"] * min(m.capacity_estimate / 1e6, 1.0)
        )

        # Penalty contributions (higher is worse)
        score -= w["drawdown_penalty"] * self._clip(m.max_drawdown_pct / 100, 0, 1)
        score -= w["turnover_penalty"] * self._clip(m.turnover, 0, 1)

        # Decay penalty: shorter half-life = worse
        if m.decay_halflife_bars < float("inf"):
            decay_norm = min(m.decay_halflife_bars / 100, 1.0)
            score -= w["decay_penalty"] * (1.0 - decay_norm)

        return round(float(score), 6)

    @staticmethod
    def _clip(value: float, lo: float, hi: float) -> float:
        """Clip a value to [lo, hi] and normalise to [0, 1]."""
        clamped = max(lo, min(hi, value))
        rng = hi - lo
        if rng <= 0:
            return 0.0
        return (clamped - lo) / rng

    # ------------------------------------------------------------------
    # Pareto frontier
    # ------------------------------------------------------------------

    def pareto_frontier(
        self,
        objectives: tuple[str, str] = ("sharpe_ratio", "max_drawdown_pct"),
    ) -> list[str]:
        """Identify alphas on the Pareto-optimal frontier.

        An alpha is Pareto-optimal if no other alpha is strictly better
        on *all* objectives simultaneously.

        Parameters
        ----------
        objectives:
            Pair of metric names.  The first is maximised, the second
            minimised.

        Returns
        -------
        list[str]
            Names of Pareto-optimal alphas.
        """
        names = list(self._entries.keys())
        values = []
        for name in names:
            m = self._entries[name].to_dict()
            values.append((m.get(objectives[0], 0.0), m.get(objectives[1], 0.0)))

        pareto: list[str] = []
        for i, (v0, v1) in enumerate(values):
            dominated = False
            for j, (u0, u1) in enumerate(values):
                if i == j:
                    continue
                # u dominates v if u has higher objective[0] AND lower objective[1]
                if u0 >= v0 and u1 <= v1 and (u0 > v0 or u1 < v1):
                    dominated = True
                    break
            if not dominated:
                pareto.append(names[i])

        logger.info(
            "Pareto frontier (%s vs %s): %d/%d alphas",
            objectives[0], objectives[1], len(pareto), len(names),
        )
        return pareto

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def top_n(self, n: int = 5) -> pd.DataFrame:
        """Return the top-N alphas by composite score.

        Parameters
        ----------
        n:
            Number of alphas to return.

        Returns
        -------
        pd.DataFrame
        """
        return self.rank().head(n)

    def clear(self) -> None:
        """Clear all entries."""
        self._entries.clear()

    @property
    def entry_count(self) -> int:
        """Number of alphas in the ranking pool."""
        return len(self._entries)
