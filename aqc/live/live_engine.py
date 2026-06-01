"""
aqc/live/live_engine.py
=========================
Async live trading engine orchestrator.

Author: Saksham Mishra — AlgoQuant Club
"""
from __future__ import annotations

import asyncio
import logging
import queue
from typing import TYPE_CHECKING, Optional

import pandas as pd

from aqc.backtester.event import EventType, MarketEvent, SignalEvent, OrderEvent, FillEvent
from aqc.backtester.event_queue import EventQueue
from aqc.live.market_data import MarketDataProvider
from aqc.live.signal_router import SignalRouter

if TYPE_CHECKING:
    from aqc.strategies.base_strategy import BaseStrategy
    from aqc.live.paper_broker import PaperBroker
    from aqc.live.portfolio_tracker import PortfolioTracker

logger = logging.getLogger(__name__)


class LiveTradingEngine:
    """Async event-driven live trading engine.

    Parameters
    ----------
    data_feed : MarketDataProvider
        Streaming live data feed.
    strategy : BaseStrategy
        Strategy to evaluate.
    portfolio_tracker : PortfolioTracker
        Real-time portfolio.
    signal_router : SignalRouter
        Routes signals and generates orders.
    paper_broker : PaperBroker
        Simulates live execution.
    event_queue : EventQueue
        Thread-safe or asyncio queue wrapper.
    """

    def __init__(
        self,
        data_feed: MarketDataProvider,
        strategy: "BaseStrategy",
        portfolio_tracker: "PortfolioTracker",
        signal_router: SignalRouter,
        paper_broker: "PaperBroker",
        event_queue: EventQueue,
    ) -> None:
        self.data_feed = data_feed
        self.strategy = strategy
        self.portfolio_tracker = portfolio_tracker
        self.signal_router = signal_router
        self.paper_broker = paper_broker
        self.event_queue = event_queue
        
        self.is_running = False
        self._loop_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the live engine."""
        self.is_running = True
        logger.info("LiveTradingEngine starting...")
        
        # Connect feed
        connected = await self.data_feed.connect()
        if not connected:
            logger.error("Failed to connect to data feed.")
            return
            
        # Start background event processing loop
        self._loop_task = asyncio.create_task(self._process_events())
        
        # Start data streaming
        try:
            async for market_event in self.data_feed.stream_bars():
                if not self.is_running:
                    break
                self.event_queue.put(market_event)
        except asyncio.CancelledError:
            logger.info("Data streaming cancelled.")
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Stop the live engine."""
        if not self.is_running:
            return
        logger.info("LiveTradingEngine stopping...")
        self.is_running = False
        await self.data_feed.disconnect()
        if self._loop_task:
            self._loop_task.cancel()

    async def _process_events(self) -> None:
        """Continuously drain the event queue."""
        while self.is_running:
            event = self.event_queue.get(block=False)
            if event is None:
                await asyncio.sleep(0.01)
                continue

            try:
                if event.event_type == EventType.MARKET:
                    self._handle_market(event)
                elif event.event_type == EventType.SIGNAL:
                    self._handle_signal(event)
                elif event.event_type == EventType.ORDER:
                    self._handle_order(event)
                elif event.event_type == EventType.FILL:
                    self._handle_fill(event)
            except Exception as e:
                logger.error("Error processing event %s: %s", event, e, exc_info=True)

    def _handle_market(self, event: MarketEvent) -> None:
        # Update portfolio MTM
        self.portfolio_tracker.portfolio.on_market_event(event)
        
        # Pass to strategy
        self.strategy.on_market_event(event)
        
        # Update broker context
        self.paper_broker.on_market_event(event)

    def _handle_signal(self, event: SignalEvent) -> None:
        curr_px = self.data_feed.get_latest_price(event.symbol) or 0.0
        order = self.signal_router.route_signal(event, curr_px)
        if order:
            self.event_queue.put(order)

    def _handle_order(self, event: OrderEvent) -> None:
        self.paper_broker.submit_order(event)

    def _handle_fill(self, event: FillEvent) -> None:
        self.portfolio_tracker.portfolio.on_fill_event(event)
        self.portfolio_tracker.snapshot()  # Save live state
