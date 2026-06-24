"""
aqc/research/scorecard/readiness_score.py
===========================================
Calculates the operational readiness score for an alpha.

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging

from aqc.alpha.alpha_base import AlphaBase

logger = logging.getLogger(__name__)


class ReadinessScorer:
    """Evaluates how ready an alpha is for deployment."""

    def calculate(self, alpha: AlphaBase, stress_score: float, robust_score: float) -> dict:
        """Calculate readiness components.

        Parameters
        ----------
        alpha:
            The alpha under test.
        stress_score:
            Score from Phase 3 stress testing (0.0 to 1.0).
        robust_score:
            Score from Phase 4 cross-market validation (0.0 to 1.0).

        Returns
        -------
        dict
            Readiness metrics.
        """
        metrics = alpha.cached_metrics
        
        # Penalties
        penalty = 0.0
        
        # Insufficient history
        if not metrics or metrics.n_signals < 1000:
            penalty += 0.2
            
        # Low capacity
        cap = getattr(metrics, "capacity_estimate", 100000) if metrics else 0
        if cap < 50000:
            penalty += 0.2

        # Base readiness based on robustness and stress survival
        base_readiness = (stress_score * 0.6) + (robust_score * 0.4)
        
        final_readiness = max(0.0, base_readiness - penalty)

        return {
            "readiness_score": final_readiness,
            "stress_score": stress_score,
            "robust_score": robust_score,
            "history_penalty": penalty > 0,
        }
