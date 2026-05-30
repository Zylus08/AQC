"""
aqc/research/regime_transitions/__init__.py
=============================================
Regime Transition Alpha Research Framework.

Determine whether regime changes themselves create alpha opportunities.

Author: Saksham Mishra — AlgoQuant Club
"""
from aqc.research.regime_transitions.transition_engine import TransitionEngine, TransitionEvent
from aqc.research.regime_transitions.transition_alpha import TransitionAlphaAnalyzer
from aqc.research.regime_transitions.transition_reports import TransitionReportGenerator
from aqc.research.regime_transitions.transition_visualization import TransitionVisualizer

__all__ = [
    "TransitionEngine",
    "TransitionEvent",
    "TransitionAlphaAnalyzer",
    "TransitionReportGenerator",
    "TransitionVisualizer",
]
