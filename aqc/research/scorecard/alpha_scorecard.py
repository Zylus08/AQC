"""
aqc/research/scorecard/alpha_scorecard.py
===========================================
Generates the Institutional Alpha Score (0-100).

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging

from aqc.alpha.alpha_base import AlphaBase
from aqc.research.scorecard.deployment_score import DeploymentScorer
from aqc.research.scorecard.readiness_score import ReadinessScorer

logger = logging.getLogger(__name__)


class AlphaScorecard:
    """Computes the final 0-100 Institutional Score."""

    def __init__(self) -> None:
        self.perf_scorer = DeploymentScorer()
        self.ready_scorer = ReadinessScorer()

    def generate_scorecard(
        self, 
        alpha: AlphaBase, 
        stress_score: float = 0.0, 
        robust_score: float = 0.0
    ) -> dict:
        """Generate the complete scorecard.

        Parameters
        ----------
        alpha:
            Evaluated AlphaBase instance.
        stress_score:
            From Phase 3 StressTestEngine (0-1).
        robust_score:
            From Phase 4 RobustnessScorer (0-1).

        Returns
        -------
        dict
            Complete scorecard metrics and final score out of 100.
        """
        perf = self.perf_scorer.calculate(alpha)
        ready = self.ready_scorer.calculate(alpha, stress_score, robust_score)

        # Final Score Formulation
        # 50% Raw Performance, 50% Operational Readiness
        final_score_raw = (perf["perf_score"] * 0.5) + (ready["readiness_score"] * 0.5)
        
        # Scale to 0-100
        institutional_score = final_score_raw * 100.0

        scorecard = {
            "alpha_name": alpha.name,
            "institutional_score": round(institutional_score, 1),
            "grade": self._assign_grade(institutional_score),
            "performance_components": perf,
            "readiness_components": ready,
            "deployable": institutional_score >= 60.0
        }
        
        logger.info("Scorecard for %s: %s/100 (Grade: %s)", 
                    alpha.name, scorecard["institutional_score"], scorecard["grade"])
                    
        return scorecard

    def _assign_grade(self, score: float) -> str:
        if score >= 90:
            return "AAA"
        elif score >= 80:
            return "AA"
        elif score >= 70:
            return "A"
        elif score >= 60:
            return "BBB"
        elif score >= 50:
            return "BB"
        else:
            return "REJECT"
