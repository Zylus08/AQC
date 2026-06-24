"""
aqc/research/generalization/robustness_score.py
=================================================
Calculates generalization and robustness scores across markets.

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class RobustnessScorer:
    """Calculates cross-market robustness score."""

    def calculate_score(self, validation_results: dict[str, dict]) -> float:
        """Calculate a 0-1.0 robustness score based on cross-market ICs.
        
        Parameters
        ----------
        validation_results:
            Output from CrossMarketValidator.run()
            
        Returns
        -------
        float
            Generalization score (0.0 to 1.0).
        """
        ics = []
        for mkt, metrics in validation_results.items():
            if "error" in metrics:
                continue
            ic = metrics.get("information_coefficient", 0.0)
            ics.append(ic)

        if not ics:
            return 0.0

        # Mean IC across domains
        mean_ic = np.mean(ics)
        # Stability across domains (lower standard deviation is better)
        std_ic = np.std(ics)
        
        # Penalize mean IC by variance
        robust_ic = mean_ic - (0.5 * std_ic)
        
        # Normalize to 0-1 (assuming max theoretical robust IC around 0.10 for high freq)
        score = np.clip(robust_ic / 0.05, 0.0, 1.0)
        
        return float(score)

    def summary_dataframe(self, results_dict: dict[str, dict]) -> pd.DataFrame:
        """Convert results dict to DataFrame."""
        df = pd.DataFrame(results_dict).T
        if "error" in df.columns:
            df = df.drop(columns=["error"])
        return df
