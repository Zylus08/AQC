"""
aqc/deployment/strategy_router.py
===================================
Strategy Router for Deployed Alphas.

Routes deployed alphas to the appropriate execution path
(e.g., Paper Trading, Small Capital, Full Deployment).

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging
from enum import Enum

from aqc.alpha.alpha_base import AlphaBase

logger = logging.getLogger(__name__)


class ExecutionTier(Enum):
    PAPER = "PAPER"
    SMALL_CAPITAL = "SMALL_CAPITAL"
    FULL_CAPITAL = "FULL_CAPITAL"


class StrategyRouter:
    """Routes deployed alphas to execution paths based on confidence."""

    def __init__(self, confidence_threshold: float = 0.8) -> None:
        self.confidence_threshold = confidence_threshold
        self._routing_map: dict[str, ExecutionTier] = {}

    def route(self, alpha: AlphaBase) -> ExecutionTier:
        """Determine execution tier for an alpha."""
        
        # In reality, this would use Bayesian Alpha Validation (from Phase 9)
        # For this skeleton, we use Sharpe + length of track record
        
        metrics = alpha.cached_metrics
        if not metrics:
            logger.warning("%s has no metrics, routing to PAPER.", alpha.name)
            tier = ExecutionTier.PAPER
        elif metrics.sharpe_ratio > 1.5 and metrics.n_signals > 500:
            tier = ExecutionTier.FULL_CAPITAL
        elif metrics.sharpe_ratio > 0.8:
            tier = ExecutionTier.SMALL_CAPITAL
        else:
            tier = ExecutionTier.PAPER

        self._routing_map[alpha.name] = tier
        logger.info("Routed %s to %s", alpha.name, tier.value)
        return tier

    def get_tier(self, alpha_name: str) -> ExecutionTier:
        return self._routing_map.get(alpha_name, ExecutionTier.PAPER)
