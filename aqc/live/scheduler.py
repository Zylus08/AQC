"""
aqc/live/scheduler.py
=======================
Periodic task scheduler for live metrics, health checks, and persistence dumps.

Author: Saksham Mishra — AlgoQuant Club
"""
from __future__ import annotations

import asyncio
import logging
from typing import Callable, Coroutine, Any

logger = logging.getLogger(__name__)


class LiveScheduler:
    """Schedules periodic async tasks during live trading."""

    def __init__(self) -> None:
        self.tasks: list[asyncio.Task] = []
        self.is_running = False

    def schedule_periodic(self, interval_seconds: float, func: Callable[..., Coroutine[Any, Any, None]], *args: Any, **kwargs: Any) -> None:
        """Register a coroutine to run every N seconds."""
        async def wrapper() -> None:
            while self.is_running:
                try:
                    await func(*args, **kwargs)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error("Error in scheduled task %s: %s", func.__name__, e, exc_info=True)
                await asyncio.sleep(interval_seconds)

        if self.is_running:
            task = asyncio.create_task(wrapper())
            self.tasks.append(task)
        else:
            # Need to queue them or just require starting first
            # We'll just bind the coroutine factory
            self._pending_wrappers = getattr(self, '_pending_wrappers', [])
            self._pending_wrappers.append(wrapper)

    def start(self) -> None:
        """Start all scheduled tasks."""
        self.is_running = True
        if hasattr(self, '_pending_wrappers'):
            for wrapper in self._pending_wrappers:
                self.tasks.append(asyncio.create_task(wrapper()))
            self._pending_wrappers.clear()

    async def stop(self) -> None:
        """Stop all tasks."""
        self.is_running = False
        for task in self.tasks:
            task.cancel()
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        self.tasks.clear()
