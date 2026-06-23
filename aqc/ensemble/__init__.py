"""
aqc/ensemble/
=============
Ensemble Alpha Engine.

Combines multiple alpha signals into a single composite alpha using
various blending and dynamic weighting methods.

Author: Saksham Mishra — AlgoQuant Club
"""

from aqc.ensemble.alpha_ensemble import AlphaEnsemble
from aqc.ensemble.model_blender import ModelBlender, BlendingMethod
from aqc.ensemble.dynamic_weighting import DynamicWeightEngine

__all__ = [
    "AlphaEnsemble",
    "ModelBlender",
    "BlendingMethod",
    "DynamicWeightEngine",
]
