"""
aqc/deployment/
===============
Deployment Pipeline Module.

Orchestrates moving alphas from the research tournament into 
live execution by passing readiness gates and allocating capital.

Author: Saksham Mishra — AlgoQuant Club
"""

from aqc.deployment.deployment_manager import DeploymentManager
from aqc.deployment.capital_allocator import CapitalAllocator
from aqc.deployment.strategy_router import StrategyRouter

__all__ = [
    "DeploymentManager",
    "CapitalAllocator",
    "StrategyRouter",
]
