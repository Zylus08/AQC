"""
aqc/research/features/feature_ranker.py
=========================================
Ranks features across multiple dimensions.

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from aqc.research.features.feature_importance import FeatureEvaluator

logger = logging.getLogger(__name__)


class FeatureRanker:
    """Ranks features combining IC and Stability."""

    def __init__(self, ic_weight: float = 0.7, stability_weight: float = 0.3) -> None:
        self.ic_weight = ic_weight
        self.stability_weight = stability_weight
        self.evaluator = FeatureEvaluator()

    def rank(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Evaluate and rank features.

        Returns
        -------
        pd.DataFrame
            Ranked leaderboard.
        """
        metrics = self.evaluator.evaluate(X, y)
        if metrics.empty:
            return metrics

        # Normalize metrics for composite score
        ic_norm = (metrics["Abs_IC"] - metrics["Abs_IC"].min()) / (metrics["Abs_IC"].max() - metrics["Abs_IC"].min() + 1e-9)
        stab_norm = (metrics["Stability"] - metrics["Stability"].min()) / (metrics["Stability"].max() - metrics["Stability"].min() + 1e-9)

        metrics["Composite_Score"] = (ic_norm * self.ic_weight) + (stab_norm * self.stability_weight)

        return metrics.sort_values("Composite_Score", ascending=False)
