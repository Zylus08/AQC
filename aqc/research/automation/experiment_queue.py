"""
aqc/research/automation/experiment_queue.py
=============================================
FIFO Queue for managing unattended experiment runs.

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging
from typing import TypedDict
from collections import deque

from aqc.research.hypothesis_lab.hypothesis_registry import HypothesisRegistry
from aqc.research.hypothesis_lab.experiment_runner import ExperimentRunner
from aqc.research.hypothesis_lab.hypothesis import HypothesisStatus

logger = logging.getLogger(__name__)


class QueuedExperiment(TypedDict):
    hypothesis_id: str
    alpha_class_name: str
    dataset_name: str


class ExperimentQueue:
    """Manages unattended execution of research experiments."""

    def __init__(self, registry: HypothesisRegistry) -> None:
        self.registry = registry
        self.runner = ExperimentRunner(registry)
        self.queue: deque[QueuedExperiment] = deque()

    def enqueue(self, hypothesis_id: str, alpha_class_name: str, dataset_name: str = "default_test") -> None:
        """Add an experiment to the queue."""
        self.queue.append({
            "hypothesis_id": hypothesis_id,
            "alpha_class_name": alpha_class_name,
            "dataset_name": dataset_name
        })
        logger.info("Queued experiment for Hypothesis: %s", hypothesis_id)

    def run_next(self, train_data, test_data) -> bool:
        """Pop and run the next experiment."""
        if not self.queue:
            logger.info("Experiment queue is empty.")
            return False
            
        exp = self.queue.popleft()
        logger.info("Running queued experiment for %s", exp["hypothesis_id"])
        
        success = self.runner.run_baseline(
            hypothesis_id=exp["hypothesis_id"],
            alpha_class_name=exp["alpha_class_name"],
            train_data=train_data,
            test_data=test_data
        )
        return success
        
    def run_all(self, train_data, test_data) -> None:
        """Run all queued experiments."""
        logger.info("Starting run of %d queued experiments.", len(self.queue))
        while self.queue:
            self.run_next(train_data, test_data)
        logger.info("Finished running experiment queue.")
