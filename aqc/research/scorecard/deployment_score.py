"""
aqc/research/scorecard/deployment_score.py
============================================
Calculates raw performance scores for deployment.

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging
import numpy as np

from aqc.alpha.alpha_base import AlphaBase

logger = logging.getLogger(__name__)


class DeploymentScorer:
    """Evaluates raw performance metrics for deployment suitability."""

    def calculate(self, alpha: AlphaBase) -> dict:
        """Calculate a 0-1 score based purely on performance.

        Parameters
        ----------
        alpha:
            The alpha under test.

        Returns
        -------
        dict
            Performance score components.
        """
        metrics = alpha.cached_metrics
        if not metrics:
            return {"perf_score": 0.0}

        # Normalize Sharpe (target 2.0+ is 1.0 score)
        sharpe_score = np.clip(metrics.sharpe_ratio / 2.0, 0.0, 1.0)
        
        # Normalize IC (target 0.05+ is 1.0 score)
        ic_score = np.clip(metrics.information_coefficient / 0.05, 0.0, 1.0)
        
        # Normalize Drawdown (target < 5% is 1.0 score, > 20% is 0 score)
        dd = metrics.max_drawdown_pct
        if dd <= 0.05:
            dd_score = 1.0
        elif dd >= 0.20:
            dd_score = 0.0
        else:
            dd_score = 1.0 - ((dd - 0.05) / 0.15)
            
        # Composite Performance Score
        perf_score = (sharpe_score * 0.4) + (ic_score * 0.4) + (dd_score * 0.2)

        return {
            "perf_score": float(perf_score),
            "sharpe_score": float(sharpe_score),
            "ic_score": float(ic_score),
            "dd_score": float(dd_score),
        }
