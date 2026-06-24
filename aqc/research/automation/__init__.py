"""
aqc/research/automation/
========================
Research Automation Engine.

Automates the execution of hypothesis tests, tournament runs,
and scorecard generation on a scheduled basis.

Author: Saksham Mishra — AlgoQuant Club
"""

from aqc.research.automation.nightly_research import NightlyResearchPipeline
from aqc.research.automation.experiment_queue import ExperimentQueue

__all__ = [
    "NightlyResearchPipeline",
    "ExperimentQueue",
]
