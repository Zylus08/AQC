"""
aqc/meta_alpha/
===============
Meta-Alpha Engine.

Intelligently selects and allocates capital across alphas based on
current market regimes and recent alpha performance.

Author: Saksham Mishra — AlgoQuant Club
"""

from aqc.meta_alpha.alpha_selector import AlphaSelector
from aqc.meta_alpha.alpha_switcher import AlphaSwitcher
from aqc.meta_alpha.allocation_optimizer import AllocationOptimizer

__all__ = [
    "AlphaSelector",
    "AlphaSwitcher",
    "AllocationOptimizer",
]
