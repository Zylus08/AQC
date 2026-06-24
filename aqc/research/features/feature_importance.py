"""
aqc/research/features/feature_importance.py
=============================================
Evaluates individual feature predictive power and stability.

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class FeatureEvaluator:
    """Evaluates features against a target."""

    def __init__(self, method: str = "IC") -> None:
        self.method = method

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Evaluate a feature matrix against targets.

        Returns
        -------
        pd.DataFrame
            Metrics per feature.
        """
        results = []
        
        # Ensure alignment
        aligned = pd.concat([X, y], axis=1).dropna()
        if aligned.empty:
            return pd.DataFrame()
            
        X_align = aligned.drop(columns=[y.name])
        y_align = aligned[y.name]

        for col in X_align.columns:
            feat = X_align[col]
            
            # Information Coefficient (Spearman rank correlation)
            ic = feat.corr(y_align, method="spearman")
            
            # Stability (auto-correlation of feature)
            stability = feat.autocorr(lag=1)
            
            # Simple Mutual Information proxy (absolute IC)
            mi_proxy = abs(ic)

            results.append({
                "Feature": col,
                "IC": ic,
                "Abs_IC": mi_proxy,
                "Stability": stability,
            })

        df = pd.DataFrame(results).set_index("Feature")
        return df.sort_values("Abs_IC", ascending=False)
