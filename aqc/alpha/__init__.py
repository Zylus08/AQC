"""
aqc/alpha/
==========
Alpha Research & Deployment infrastructure.

Provides the universal alpha interface, registry, factory, ranking,
monitoring, and lifecycle management for all AQC alpha signals.

Author: Saksham Mishra — AlgoQuant Club
"""

from aqc.alpha.alpha_base import (
    AlphaBase,
    AlphaSignal,
    AlphaMetrics,
    AlphaMetadata,
    AlphaCategory,
    AlphaFrequency,
)
from aqc.alpha.alpha_registry import AlphaRegistry, register_alpha
from aqc.alpha.alpha_factory import AlphaFactory
from aqc.alpha.alpha_ranker import AlphaRanker
from aqc.alpha.alpha_monitor import AlphaMonitor
from aqc.alpha.alpha_decay_monitor import AlphaDecayMonitor

__all__ = [
    "AlphaBase",
    "AlphaSignal",
    "AlphaMetrics",
    "AlphaMetadata",
    "AlphaCategory",
    "AlphaFrequency",
    "AlphaRegistry",
    "register_alpha",
    "AlphaFactory",
    "AlphaRanker",
    "AlphaMonitor",
    "AlphaDecayMonitor",
]
