"""
aqc/research/automation/alpha_scheduler.py
============================================
Schedules research tasks.

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging
from typing import Callable

logger = logging.getLogger(__name__)


class AlphaScheduler:
    """Mock scheduler for running nightly jobs.
    
    In production, this would wrap APScheduler or Celery Beat.
    """

    def __init__(self) -> None:
        self.jobs: dict[str, Callable] = {}

    def add_daily_job(self, name: str, func: Callable, time_str: str) -> None:
        """Add a job to run daily at specific time (e.g. '00:00')."""
        self.jobs[name] = func
        logger.info("Scheduled job '%s' to run daily at %s", name, time_str)

    def trigger(self, name: str) -> None:
        """Manually trigger a scheduled job."""
        if name in self.jobs:
            logger.info("Triggering scheduled job: %s", name)
            self.jobs[name]()
        else:
            logger.error("Job %s not found.", name)
