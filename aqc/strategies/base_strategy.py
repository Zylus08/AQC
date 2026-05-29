"""
aqc/strategies/base_strategy.py
================================
Abstract base class for all AQC trading strategies.

Design Contract
---------------
Every strategy MUST:

1. Inherit from :class:`BaseStrategy`.
2. Implement :meth:`generate_signal` to encapsulate the entry/exit logic.
3. Implement :meth:`on_market_event` as the primary event hook.

Strategies communicate with the rest of the system *exclusively* through
the shared event queue — they never call portfolio or broker methods directly.
This ensures clean decoupling between signal generation and order management.

Strategy State
--------------
Each strategy maintains an internal ``_bar_buffer`` — a rolling window of
recent bars per symbol used to compute indicators without replaying the
entire history on every bar.

Author: AQC Team
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Optional

import pandas as pd

from aqc.backtester.event import MarketEvent, SignalEvent

if TYPE_CHECKING:
    from aqc.backtester.event_queue import EventQueue

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """Abstract base class for all AQC strategies.

    Parameters
    ----------
    event_queue:
        Shared event queue.  Strategies enqueue
        :class:`~aqc.backtester.event.SignalEvent` objects here.
    symbols:
        List of instrument tickers this strategy trades.
    strategy_id:
        Human-readable name used for logging and PnL attribution.
    lookback:
        Number of bars to keep in the rolling buffer per symbol.
        Set this to at least the maximum indicator period used by the
        strategy.

    Attributes
    ----------
    _bar_buffer:
        ``defaultdict(deque)`` — a rolling window of the last *lookback*
        :class:`~aqc.backtester.event.MarketEvent` objects per symbol.
    _bar_dataframes:
        ``defaultdict(pd.DataFrame)`` — a cached DataFrame view of the
        buffer for vectorised indicator calculations.
    _signal_count:
        Running count of signals emitted (for diagnostics).
    """

    def __init__(
        self,
        event_queue: "EventQueue",
        symbols: list[str],
        strategy_id: str = "base_strategy",
        lookback: int = 200,
    ) -> None:
        self._eq = event_queue
        self.symbols = symbols
        self.strategy_id = strategy_id
        self.lookback = lookback

        self._bar_buffer: dict[str, deque[MarketEvent]] = defaultdict(
            lambda: deque(maxlen=lookback)
        )
        self._bar_dataframes: dict[str, pd.DataFrame] = {}
        self._signal_count: int = 0

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def generate_signal(self, symbol: str, bars: pd.DataFrame) -> Optional[SignalEvent]:
        """Compute a trading signal from the current bar window.

        This is the primary method subclasses must implement.  It is called
        by :meth:`on_market_event` after the buffer is updated.

        Parameters
        ----------
        symbol:
            The instrument for which to generate a signal.
        bars:
            DataFrame of OHLCV data for the last *lookback* bars, with
            columns ``open``, ``high``, ``low``, ``close``, ``volume``.

        Returns
        -------
        SignalEvent | None
            A signal if entry/exit conditions are met, or ``None`` to pass.
        """

    # ------------------------------------------------------------------
    # Event hook
    # ------------------------------------------------------------------

    def on_market_event(self, event: MarketEvent) -> None:
        """Handle an incoming market bar and potentially emit a signal.

        This method:

        1. Checks whether this strategy trades the event's symbol.
        2. Appends the bar to the rolling buffer.
        3. Refreshes the DataFrame cache.
        4. Calls :meth:`generate_signal` when the buffer has enough bars.
        5. Enqueues any returned signal.

        Parameters
        ----------
        event:
            The incoming market bar.
        """
        if event.symbol not in self.symbols:
            return

        symbol = event.symbol
        self._bar_buffer[symbol].append(event)
        self._refresh_dataframe(symbol)

        bars = self._bar_dataframes[symbol]

        if len(bars) < self.min_bars_required:
            logger.debug(
                "%s: only %d/%d bars for %s — warming up",
                self.strategy_id,
                len(bars),
                self.min_bars_required,
                symbol,
            )
            return

        signal = self.generate_signal(symbol, bars)
        if signal is not None:
            self._emit_signal(signal)

    # ------------------------------------------------------------------
    # Configuration hooks (override if needed)
    # ------------------------------------------------------------------

    @property
    def min_bars_required(self) -> int:
        """Minimum number of bars before signal generation starts.

        Override in subclasses to specify a warm-up period.

        Returns
        -------
        int
            Default is 2 (enough for a single-bar comparison).
        """
        return 2

    # ------------------------------------------------------------------
    # Signal emission
    # ------------------------------------------------------------------

    def _emit_signal(self, signal: SignalEvent) -> None:
        """Enqueue a signal event and update diagnostic counters.

        Parameters
        ----------
        signal:
            The signal to emit.
        """
        self._eq.put(signal)
        self._signal_count += 1
        logger.debug(
            "%s signal #%d: %s %s strength=%.2f",
            self.strategy_id,
            self._signal_count,
            signal.direction.value,
            signal.symbol,
            signal.strength,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _refresh_dataframe(self, symbol: str) -> None:
        """Rebuild the cached DataFrame from the bar buffer.

        This is called after every new bar is appended.

        Parameters
        ----------
        symbol:
            Symbol to refresh.
        """
        buffer = self._bar_buffer[symbol]
        self._bar_dataframes[symbol] = pd.DataFrame(
            [
                {
                    "timestamp": e.bar_time,
                    "open": e.open_price,
                    "high": e.high_price,
                    "low": e.low_price,
                    "close": e.close_price,
                    "volume": e.volume,
                }
                for e in buffer
            ]
        ).set_index("timestamp")

    def get_bars(self, symbol: str) -> pd.DataFrame:
        """Return the current bar window for a symbol.

        Parameters
        ----------
        symbol:
            Instrument ticker.

        Returns
        -------
        pd.DataFrame
            OHLCV DataFrame, or empty DataFrame if no data yet.
        """
        return self._bar_dataframes.get(symbol, pd.DataFrame())

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"id={self.strategy_id}, "
            f"symbols={self.symbols}, "
            f"signals_emitted={self._signal_count})"
        )
