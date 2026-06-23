"""
aqc/deployment/capital_allocator.py
=====================================
Capital Allocator for Deployed Alphas.

Determines the monetary allocation for each deployed alpha based on
risk-parity or volatility targeting.

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging

import pandas as pd

from aqc.alpha.alpha_base import AlphaBase

logger = logging.getLogger(__name__)


class CapitalAllocator:
    """Allocates capital among deployed alphas.

    Parameters
    ----------
    total_capital:
        Total base currency capital to allocate.
    max_allocation_pct:
        Maximum percentage of total capital allocated to a single alpha.
    """

    def __init__(
        self,
        total_capital: float = 1000000.0,
        max_allocation_pct: float = 0.30,
    ) -> None:
        self.total_capital = total_capital
        self.max_allocation_pct = max_allocation_pct

    def allocate_equal(self, deployed_alphas: list[AlphaBase]) -> dict[str, float]:
        """Equal weight allocation."""
        if not deployed_alphas:
            return {}
        
        n = len(deployed_alphas)
        weight = min(1.0 / n, self.max_allocation_pct)
        # Scale if capping reduces total below 1.0
        # In a real system, we'd distribute the remainder. Keeping it simple here.
        allocation = self.total_capital * weight
        
        return {a.name: allocation for a in deployed_alphas}

    def allocate_risk_parity(self, deployed_alphas: list[AlphaBase]) -> dict[str, float]:
        """Risk parity allocation (inverse volatility/drawdown proxy)."""
        if not deployed_alphas:
            return {}

        inv_risk = {}
        total_inv_risk = 0.0

        for a in deployed_alphas:
            # Use max drawdown as risk proxy if actual vol isn't easily accessible
            dd = a.cached_metrics.max_drawdown_pct if a.cached_metrics else 10.0
            dd = max(1e-4, dd)
            r = 1.0 / dd
            inv_risk[a.name] = r
            total_inv_risk += r

        allocations = {}
        for name, ir in inv_risk.items():
            weight = ir / total_inv_risk
            weight = min(weight, self.max_allocation_pct)
            allocations[name] = self.total_capital * weight

        return allocations
