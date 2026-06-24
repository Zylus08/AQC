"""
aqc/research/alpha_decay/
=========================
Alpha Decay Research.

Formally manages the end-of-life process for alphas
that have structurally decayed.

Author: Saksham Mishra — AlgoQuant Club
"""

from aqc.research.alpha_decay.decay_detector import DecayDetector
from aqc.research.alpha_decay.alpha_lifecycle import LifecycleManager

__all__ = [
    "DecayDetector",
    "LifecycleManager",
]
