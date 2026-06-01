"""
aqc/live/health_monitor.py
============================
System health monitoring: checks feed latency, heartbeat, and order errors.

Author: Saksham Mishra — AlgoQuant Club
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
import time

import pandas as pd

logger = logging.getLogger(__name__)


class HealthState(Enum):
    HEALTHY = "HEALTHY"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class HealthStatus:
    timestamp: pd.Timestamp
    state: HealthState
    feed_latency_ms: float
    last_heartbeat_s: float
    stale_data_warning: bool
    rejected_orders_count: int
    message: str


class HealthMonitor:
    """Monitors live system health.

    Parameters
    ----------
    stale_threshold_s : float
        Seconds before feed is considered stale.
    """

    def __init__(self, stale_threshold_s: float = 300.0) -> None:
        self.stale_threshold_s = stale_threshold_s
        self.last_market_time: float = time.time()
        self.rejected_orders = 0
        self.status_history: list[HealthStatus] = []

    def ping_feed(self) -> None:
        self.last_market_time = time.time()

    def record_rejection(self) -> None:
        self.rejected_orders += 1

    def check_health(self) -> HealthStatus:
        now = time.time()
        time_since_last = now - self.last_market_time
        
        state = HealthState.HEALTHY
        msg = "System operating normally."
        stale = False
        
        if time_since_last > self.stale_threshold_s:
            state = HealthState.WARNING
            msg = f"Feed stale for {time_since_last:.0f} seconds."
            stale = True
            
        if self.rejected_orders > 10:
            state = HealthState.CRITICAL
            msg = f"High order rejection rate ({self.rejected_orders} rejections)."

        status = HealthStatus(
            timestamp=pd.Timestamp.utcnow(),
            state=state,
            feed_latency_ms=time_since_last * 1000 if not stale else 0.0,
            last_heartbeat_s=time_since_last,
            stale_data_warning=stale,
            rejected_orders_count=self.rejected_orders,
            message=msg
        )
        
        self.status_history.append(status)
        return status
