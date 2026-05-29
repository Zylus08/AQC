"""
aqc/backtester/engine.py
========================
Core event-driven backtest engine.

The :class:`BacktestEngine` orchestrates the full simulation loop:

1. Iterates over market data bar-by-bar.
2. Emits a :class:`~aqc.backtester.event.MarketEvent` for each bar.
3. Drains the event queue until empty (processes SIGNAL, ORDER, FILL events).
4. Advances to the next bar.
5. After all bars are processed, generates a performance report.

This architecture guarantees *no look-ahead bias* because the strategy only
sees data up to and including the current bar.

Author: AQC Team
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import TYPE_CHECKING, Optional

import pandas as pd

from aqc.backtester.event import EventType, MarketEvent
from aqc.backtester.event_queue import EventQueue
from aqc.backtester.execution import ExecutionEngine
from aqc.backtester.portfolio import Portfolio
from aqc.analytics.metrics import PerformanceMetrics
from aqc.analytics.reporting import ReportGenerator

if TYPE_CHECKING:
    from aqc.strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Coordinates the full event-driven backtesting simulation.

    Parameters
    ----------
    data:
        Dictionary mapping ``symbol → OHLCV DataFrame``.  Each DataFrame
        must have a :class:`~pandas.DatetimeIndex` and columns
        ``open``, ``high``, ``low``, ``close``, ``volume``.
    strategy:
        Instantiated strategy that will receive market events.
    portfolio:
        Instantiated portfolio that tracks positions and PnL.
    execution_engine:
        Execution engine wrapping the simulated (or live) broker.
    event_queue:
        Shared event queue; all components read from and write to this.
    config:
        Optional configuration dictionary with run-time parameters.

    Examples
    --------
    >>> engine = BacktestEngine(
    ...     data={"AAPL": ohlcv_df},
    ...     strategy=my_strategy,
    ...     portfolio=portfolio,
    ...     execution_engine=execution_engine,
    ...     event_queue=event_queue,
    ... )
    >>> results = engine.run()
    """

    def __init__(
        self,
        data: dict[str, pd.DataFrame],
        strategy: "BaseStrategy",
        portfolio: Portfolio,
        execution_engine: ExecutionEngine,
        event_queue: EventQueue,
        config: Optional[dict] = None,
    ) -> None:
        self.data = data
        self.strategy = strategy
        self.portfolio = portfolio
        self.execution_engine = execution_engine
        self.event_queue = event_queue
        self.config = config or {}

        self._running = False
        self._bar_count = 0
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None

        # Align all symbols to a common time index
        self._common_index = self._build_common_index()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """Execute the full backtest simulation.

        Returns
        -------
        dict
            Results dictionary containing portfolio summary, performance
            metrics, equity curve, and trade log.
        """
        logger.info("=" * 60)
        logger.info("AQC BacktestEngine — simulation starting")
        logger.info("  Symbols   : %s", list(self.data.keys()))
        logger.info("  Bars      : %d", len(self._common_index))
        logger.info("  Capital   : %.2f", self.portfolio.initial_capital)
        logger.info("=" * 60)

        self._running = True
        self._start_time = time.perf_counter()

        for bar_time in self._common_index:
            self._process_bar(bar_time)

        self._end_time = time.perf_counter()
        self._running = False

        elapsed = self._end_time - self._start_time
        logger.info(
            "Simulation complete: %d bars processed in %.3f s (%.0f bars/s)",
            self._bar_count,
            elapsed,
            self._bar_count / max(elapsed, 1e-9),
        )

        return self._compile_results()

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def _process_bar(self, bar_time: pd.Timestamp) -> None:
        """Process a single bar across all symbols.

        Steps
        -----
        1. Emit a MarketEvent for each symbol with data at this bar.
        2. Notify the strategy, portfolio mark-to-market, and execution cache.
        3. Drain the event queue (handles SIGNALs, ORDERs, FILLs).

        Parameters
        ----------
        bar_time:
            The timestamp of the current bar.
        """
        for symbol, df in self.data.items():
            if bar_time not in df.index:
                continue

            row = df.loc[bar_time]
            market_event = self._build_market_event(symbol, bar_time, row)

            # 1. Update execution engine's price cache
            self.execution_engine.on_market_event(market_event)

            # 2. Mark portfolio to market
            self.portfolio.on_market_event(market_event)

            # 3. Notify strategy — may enqueue SignalEvent(s)
            self.strategy.on_market_event(market_event)

        # 4. Drain the event queue for this bar
        self._drain_queue()
        self._bar_count += 1

    def _drain_queue(self) -> None:
        """Process all events currently on the queue.

        The loop continues until the queue is empty.  Each event type is
        routed to the correct handler:

        * ``SIGNAL`` → :meth:`~aqc.backtester.portfolio.Portfolio.on_signal_event`
        * ``ORDER``  → :meth:`~aqc.backtester.execution.ExecutionEngine.on_order_event`
        * ``FILL``   → :meth:`~aqc.backtester.portfolio.Portfolio.on_fill_event`
        * ``MARKET`` → ignored (already processed above)
        """
        max_iterations = 10_000  # guard against infinite loops
        iteration = 0

        while not self.event_queue.empty() and iteration < max_iterations:
            event = self.event_queue.get(block=False)
            if event is None:
                break

            if event.event_type == EventType.SIGNAL:
                self.portfolio.on_signal_event(event)  # type: ignore[arg-type]

            elif event.event_type == EventType.ORDER:
                self.execution_engine.on_order_event(event)  # type: ignore[arg-type]

            elif event.event_type == EventType.FILL:
                self.portfolio.on_fill_event(event)  # type: ignore[arg-type]

            elif event.event_type == EventType.MARKET:
                pass  # already handled above

            else:
                logger.warning("Unknown event type: %s", event.event_type)

            iteration += 1

        if iteration >= max_iterations:
            logger.error(
                "Event drain hit iteration limit (%d) — possible event loop bug.",
                max_iterations,
            )

    # ------------------------------------------------------------------
    # Results compilation
    # ------------------------------------------------------------------

    def _compile_results(self) -> dict:
        """Aggregate portfolio state and compute performance metrics.

        Returns
        -------
        dict
            Comprehensive results dictionary.
        """
        port_summary = self.portfolio.summary()

        # Build equity curve DataFrame
        eq_curve = pd.DataFrame(
            [
                {
                    "timestamp": snap.timestamp,
                    "equity": snap.equity,
                    "cash": snap.cash,
                    "total_pnl": snap.total_pnl,
                }
                for snap in self.portfolio.equity_curve
            ]
        )

        if not eq_curve.empty:
            eq_curve.set_index("timestamp", inplace=True)

        # Compute performance metrics
        metrics = PerformanceMetrics(equity_curve=eq_curve, trade_log=self.portfolio.trade_log)
        perf = metrics.compute_all()

        # Generate text report
        reporter = ReportGenerator(
            portfolio_summary=port_summary,
            performance_metrics=perf,
            equity_curve=eq_curve,
            trade_log=self.portfolio.trade_log,
        )
        reporter.print_report()

        elapsed = (self._end_time or 0.0) - (self._start_time or 0.0)

        return {
            "portfolio_summary": port_summary,
            "performance_metrics": perf,
            "equity_curve": eq_curve,
            "trade_log": self.portfolio.trade_log,
            "event_queue_stats": self.event_queue.stats,
            "elapsed_seconds": round(elapsed, 4),
            "bars_processed": self._bar_count,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_common_index(self) -> pd.DatetimeIndex:
        """Create a sorted union of all symbols' timestamps.

        Returns
        -------
        pd.DatetimeIndex
            Sorted union of all bar timestamps across all symbols.
        """
        indices = [df.index for df in self.data.values()]
        if not indices:
            return pd.DatetimeIndex([])
        combined = indices[0]
        for idx in indices[1:]:
            combined = combined.union(idx)
        return combined.sort_values()

    @staticmethod
    def _build_market_event(
        symbol: str, bar_time: pd.Timestamp, row: pd.Series
    ) -> MarketEvent:
        """Construct a :class:`~aqc.backtester.event.MarketEvent` from a DataFrame row.

        Parameters
        ----------
        symbol:
            Instrument ticker.
        bar_time:
            Bar timestamp.
        row:
            OHLCV Series for the current bar.

        Returns
        -------
        MarketEvent
        """
        return MarketEvent(
            symbol=symbol,
            bar_time=bar_time.to_pydatetime(),
            open_price=float(row.get("open", row.get("Open", 0.0))),
            high_price=float(row.get("high", row.get("High", 0.0))),
            low_price=float(row.get("low", row.get("Low", 0.0))),
            close_price=float(row.get("close", row.get("Close", 0.0))),
            volume=float(row.get("volume", row.get("Volume", 0.0))),
            vwap=float(row["vwap"]) if "vwap" in row.index else None,
        )

    def __repr__(self) -> str:
        return (
            f"BacktestEngine("
            f"symbols={list(self.data.keys())}, "
            f"bars={len(self._common_index)}, "
            f"running={self._running})"
        )
