"""
aqc/live/alert_manager.py
===========================
Generates operational alerts for risk violations, feed disconnects, etc.

Author: Saksham Mishra — AlgoQuant Club
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import pandas as pd

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class Alert:
    timestamp: pd.Timestamp
    level: AlertLevel
    source: str
    message: str


class AlertManager:
    """Manages system alerts.

    Currently supports console and logs.
    Future: Email, Telegram, Discord webhooks.
    """

    def __init__(self) -> None:
        self.alerts: list[Alert] = []

    def trigger(self, level: AlertLevel, source: str, message: str) -> None:
        """Trigger a new alert."""
        alert = Alert(pd.Timestamp.utcnow(), level, source, message)
        self.alerts.append(alert)
        
        log_msg = f"[ALERT - {level.value}] {source}: {message}"
        if level == AlertLevel.CRITICAL:
            logger.critical(log_msg)
        elif level == AlertLevel.WARNING:
            logger.warning(log_msg)
        else:
            logger.info(log_msg)
            
        # Future webhook implementations here

    def get_recent_alerts(self, limit: int = 10) -> list[Alert]:
        return self.alerts[-limit:]
