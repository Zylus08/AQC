"""
aqc/research/generalization/
============================
Cross-Market Validation Engine.

Tests alphas across multiple asset classes to determine generalization robustness.

Author: Saksham Mishra — AlgoQuant Club
"""

from aqc.research.generalization.market_validator import CrossMarketValidator
from aqc.research.generalization.robustness_score import RobustnessScorer

__all__ = [
    "CrossMarketValidator",
    "RobustnessScorer",
]
