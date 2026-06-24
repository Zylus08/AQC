"""
aqc/live_capital/deployment_guard.py
======================================
Guard checks before an alpha is allowed live capital.

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging

from aqc.research.scorecard.alpha_scorecard import AlphaScorecard
from aqc.alpha.alpha_base import AlphaBase

logger = logging.getLogger(__name__)


class DeploymentGuard:
    """Blocks alphas that do not meet institutional readiness scores."""

    def __init__(self, min_institutional_score: float = 60.0) -> None:
        self.min_score = min_institutional_score
        self.scorecard_gen = AlphaScorecard()

    def check(self, alpha: AlphaBase, stress_score: float, robust_score: float) -> bool:
        """Evaluate if alpha can be deployed to live capital."""
        scorecard = self.scorecard_gen.generate_scorecard(alpha, stress_score, robust_score)
        
        score = scorecard["institutional_score"]
        if score >= self.min_score:
            logger.info("GUARD PASSED: %s (Score: %.1f)", alpha.name, score)
            return True
        else:
            logger.warning("GUARD BLOCKED: %s (Score: %.1f < %.1f)", alpha.name, score, self.min_score)
            return False
