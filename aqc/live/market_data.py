"""
aqc/live/market_data.py
=========================
Abstract interface for live market data providers.

Author: Saksham Mishra — AlgoQuant Club
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional

from aqc.backtester.event import MarketEvent

logger = logging.getLogger(__name__)


class MarketDataProvider(ABC):
    """Abstract interface for live streaming market data feeds."""

    def __init__(self, symbols: list[str]) -> None:
        self.symbols = symbols
        self.is_connected = False

    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to the data source."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the data source."""

    @abstractmethod
    async def stream_bars(self) -> AsyncGenerator[MarketEvent, None]:
        """Async generator yielding MarketEvents as they arrive."""

    @abstractmethod
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Return the most recently seen price for a symbol."""
