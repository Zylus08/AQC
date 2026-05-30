"""
aqc/live/live_data_feed.py
============================
Simulated and Yahoo Finance live data feeds.

Author: Saksham Mishra — AlgoQuant Club
"""
from __future__ import annotations

import asyncio
import logging
from typing import AsyncGenerator, Optional

import pandas as pd
import yfinance as yf

from aqc.backtester.event import MarketEvent
from aqc.live.market_data import MarketDataProvider

logger = logging.getLogger(__name__)


class SimulatedFeed(MarketDataProvider):
    """Replays historical DataFrame as a live stream using asyncio.sleep.

    Parameters
    ----------
    data : dict[str, pd.DataFrame]
        Historical OHLCV data.
    interval_seconds : float
        Simulated delay between bars (e.g., 0.1 for fast replay).
    """

    def __init__(self, data: dict[str, pd.DataFrame], interval_seconds: float = 1.0) -> None:
        super().__init__(list(data.keys()))
        self.data = data
        self.interval = interval_seconds
        self._latest_prices: dict[str, float] = {}

    async def connect(self) -> bool:
        self.is_connected = True
        logger.info("SimulatedFeed connected.")
        return True

    async def disconnect(self) -> None:
        self.is_connected = False
        logger.info("SimulatedFeed disconnected.")

    async def stream_bars(self) -> AsyncGenerator[MarketEvent, None]:
        if not self.is_connected:
            return

        # Align all symbols to a common index to simulate time progressing
        all_timestamps = set()
        for df in self.data.values():
            all_timestamps.update(df.index)
        
        timeline = sorted(list(all_timestamps))

        for ts in timeline:
            if not self.is_connected:
                break
                
            for symbol in self.symbols:
                df = self.data[symbol]
                if ts in df.index:
                    row = df.loc[ts]
                    close_px = float(row.get("close", 0.0))
                    self._latest_prices[symbol] = close_px
                    
                    event = MarketEvent(
                        symbol=symbol,
                        timestamp=pd.Timestamp(ts),
                        open_price=float(row.get("open", close_px)),
                        high_price=float(row.get("high", close_px)),
                        low_price=float(row.get("low", close_px)),
                        close_price=close_px,
                        volume=float(row.get("volume", 0.0)),
                    )
                    yield event
                    
            await asyncio.sleep(self.interval)

    def get_latest_price(self, symbol: str) -> Optional[float]:
        return self._latest_prices.get(symbol)


class YahooFinanceFeed(MarketDataProvider):
    """Fetches real-time snapshots from Yahoo Finance via polling.

    Parameters
    ----------
    symbols : list[str]
        Tickers to poll.
    poll_interval : float
        Polling interval in seconds (default 60s).
    """

    def __init__(self, symbols: list[str], poll_interval: float = 60.0) -> None:
        super().__init__(symbols)
        self.poll_interval = poll_interval
        self._latest_prices: dict[str, float] = {}

    async def connect(self) -> bool:
        self.is_connected = True
        logger.info("YahooFinanceFeed connected.")
        return True

    async def disconnect(self) -> None:
        self.is_connected = False
        logger.info("YahooFinanceFeed disconnected.")

    async def stream_bars(self) -> AsyncGenerator[MarketEvent, None]:
        if not self.is_connected:
            return

        while self.is_connected:
            try:
                # yf.download is blocking, so we run it in a thread executor
                loop = asyncio.get_running_loop()
                data = await loop.run_in_executor(None, self._fetch_data)

                if data is not None and not data.empty:
                    ts = pd.Timestamp.utcnow()
                    for symbol in self.symbols:
                        if symbol in data.columns.levels[1]:
                            row = data.xs(symbol, level=1, axis=1).iloc[-1]
                            close_px = float(row.get("Close", 0.0))
                            self._latest_prices[symbol] = close_px
                            
                            event = MarketEvent(
                                symbol=symbol,
                                timestamp=ts,
                                open_price=float(row.get("Open", close_px)),
                                high_price=float(row.get("High", close_px)),
                                low_price=float(row.get("Low", close_px)),
                                close_price=close_px,
                                volume=float(row.get("Volume", 0.0)),
                            )
                            yield event
            except Exception as e:
                logger.error("YahooFinanceFeed error: %s", e)
            
            await asyncio.sleep(self.poll_interval)

    def _fetch_data(self) -> Optional[pd.DataFrame]:
        try:
            return yf.download(
                self.symbols, 
                period="1d", 
                interval="1m", 
                progress=False, 
                group_by="ticker"
            )
        except Exception:
            return None

    def get_latest_price(self, symbol: str) -> Optional[float]:
        return self._latest_prices.get(symbol)
