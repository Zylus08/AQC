"""
aqc/research/stress_testing/parameter_stability.py
====================================================
Tests alpha robustness against parameter perturbations.

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging
import copy
from typing import Any

import pandas as pd
import numpy as np

from aqc.alpha.alpha_base import AlphaBase

logger = logging.getLogger(__name__)


class ParameterStabilityTester:
    """Tests alpha performance stability across parameter spaces."""

    def __init__(self, steps: int = 5, variance_pct: float = 0.20) -> None:
        self.steps = steps
        self.variance_pct = variance_pct

    def run(self, alpha: AlphaBase, param_name: str, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Perturb a parameter and evaluate the alpha.
        
        Requires the alpha to have the parameter as an attribute.
        """
        if not hasattr(alpha, param_name):
            logger.error("Alpha %s does not have parameter %s", alpha.name, param_name)
            return pd.DataFrame()

        base_val = getattr(alpha, param_name)
        if not isinstance(base_val, (int, float)):
            logger.error("Parameter %s must be numeric.", param_name)
            return pd.DataFrame()

        # Generate linear space of values around the base value
        min_val = base_val * (1.0 - self.variance_pct)
        max_val = base_val * (1.0 + self.variance_pct)
        test_vals = np.linspace(min_val, max_val, self.steps)
        
        if isinstance(base_val, int):
            test_vals = np.unique(np.round(test_vals).astype(int))

        results = []
        logger.info("Testing %s on %s values: %s", param_name, len(test_vals), test_vals)

        for val in test_vals:
            # Create a clone to avoid mutating the original
            test_alpha = copy.deepcopy(alpha)
            setattr(test_alpha, param_name, val)
            
            # Predict and evaluate
            preds = test_alpha.predict(X)
            metrics = test_alpha.evaluate(preds, y)
            
            results.append({
                "Parameter_Value": val,
                "Sharpe": metrics.sharpe_ratio,
                "IC": metrics.information_coefficient,
            })

        df = pd.DataFrame(results)
        
        # Calculate stability score (lower variance is better)
        sharpe_std = df["Sharpe"].std()
        logger.info("Parameter %s stability complete. Sharpe StdDev: %.4f", param_name, sharpe_std)
        
        return df
