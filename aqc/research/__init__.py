"""
aqc/research/__init__.py
========================
Research infrastructure — Walk-Forward Optimization, Parameter Search,
and Out-of-Sample Validation.

Author: Saksham Mishra — AlgoQuant Club
"""

from aqc.research.parameter_space import (
    ParameterSpace,
    IntParam,
    FloatParam,
    CategoricalParam,
    ParameterGrid,
)
from aqc.research.optimizer import (
    GridSearchOptimizer,
    RandomSearchOptimizer,
    OptimizationResult,
    ObjectiveMetric,
)
from aqc.research.walk_forward import (
    WalkForwardEngine,
    WalkForwardMode,
    WalkForwardFold,
    WalkForwardResult,
)
from aqc.research.validation import WalkForwardValidator

__all__ = [
    # Parameter space
    "ParameterSpace",
    "IntParam",
    "FloatParam",
    "CategoricalParam",
    "ParameterGrid",
    # Optimizer
    "GridSearchOptimizer",
    "RandomSearchOptimizer",
    "OptimizationResult",
    "ObjectiveMetric",
    # Walk-forward engine
    "WalkForwardEngine",
    "WalkForwardMode",
    "WalkForwardFold",
    "WalkForwardResult",
    # Validation
    "WalkForwardValidator",
]
