"""
aqc/portfolio/__init__.py
==========================
Volatility-Targeted Portfolio Construction.

* :class:`VolatilityTargetedPortfolio` — Vol-forecast-aware position sizer
* :class:`PortfolioAllocator` — Multi-asset allocation with constraints
* Portfolio-level risk metrics (VaR, ES, turnover, concentration)

Author: Saksham Mishra — AlgoQuant Club
"""

from aqc.portfolio.volatility_portfolio import VolatilityTargetedPortfolio
from aqc.portfolio.allocation import PortfolioAllocator, AllocationConstraints
from aqc.portfolio.portfolio_metrics import PortfolioRiskMetrics

__all__ = [
    "VolatilityTargetedPortfolio",
    "PortfolioAllocator",
    "AllocationConstraints",
    "PortfolioRiskMetrics",
]
