"""
aqc/research/scorecard/
=======================
Institutional Alpha Scorecard.

Generates a unified 0-100 score for alpha strategies based on
returns, capacity, stress tests, and deployment readiness.

Author: Saksham Mishra — AlgoQuant Club
"""

from aqc.research.scorecard.alpha_scorecard import AlphaScorecard
from aqc.research.scorecard.deployment_score import DeploymentScorer
from aqc.research.scorecard.readiness_score import ReadinessScorer

__all__ = [
    "AlphaScorecard",
    "DeploymentScorer",
    "ReadinessScorer",
]
