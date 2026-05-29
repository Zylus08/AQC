"""
aqc/backtester/event_queue.py
=============================
Thread-safe priority event queue for the AQC backtesting engine.

Design Notes
------------
* Built on top of Python's :class:`queue.Queue` (unbounded, thread-safe FIFO).
* Events carry an explicit :class:`~aqc.backtester.event.EventType`; the
  engine processes them in the order they are placed on the queue, which
  mirrors the causal chain:

      MARKET → SIGNAL → ORDER → FILL

* A separate *priority* channel is provided for system-level control
  messages (e.g. engine shutdown) via :meth:`put_priority`.

Author: AQC Team
"""

from __future__ import annotations

import queue
import threading
from typing import Optional

from aqc.backtester.event import BaseEvent


class EventQueue:
    """A thread-safe FIFO queue that transports :class:`~aqc.backtester.event.BaseEvent` objects.

    The queue is the single shared data structure between all engine
    components.  Producers call :meth:`put`; the engine's main loop calls
    :meth:`get` in a tight loop until the queue is drained for each bar.

    Attributes
    ----------
    _queue:
        Internal :class:`queue.Queue` instance.
    _lock:
        Re-entrant lock protecting queue-level statistics (size snapshots,
        counters).  The underlying :class:`queue.Queue` is already
        thread-safe, but the lock allows atomic read-then-write operations.

    Examples
    --------
    >>> from aqc.backtester.event_queue import EventQueue
    >>> from aqc.backtester.event import MarketEvent
    >>> eq = EventQueue()
    >>> eq.put(MarketEvent(symbol="AAPL", close_price=150.0))
    >>> event = eq.get(block=False)
    >>> event.symbol
    'AAPL'
    """

    def __init__(self) -> None:
        self._queue: queue.Queue[BaseEvent] = queue.Queue()
        self._lock: threading.RLock = threading.RLock()
        self._total_enqueued: int = 0
        self._total_dequeued: int = 0

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def put(self, event: BaseEvent) -> None:
        """Enqueue an event.

        Parameters
        ----------
        event:
            Any :class:`~aqc.backtester.event.BaseEvent` subclass instance.

        Raises
        ------
        TypeError:
            If *event* is not an instance of :class:`~aqc.backtester.event.BaseEvent`.
        """
        if not isinstance(event, BaseEvent):
            raise TypeError(
                f"EventQueue only accepts BaseEvent subclasses, got {type(event)}"
            )
        with self._lock:
            self._total_enqueued += 1
        self._queue.put_nowait(event)

    def put_priority(self, event: BaseEvent) -> None:
        """Place an event at the *front* of the queue using an internal trick.

        .. warning::
            :class:`queue.Queue` does not natively support priority insertion.
            This method works by temporarily draining the queue, inserting the
            priority event, then re-inserting the drained items.  It is
            intended for low-frequency control events only (e.g. shutdown
            signals) and must **not** be called on the hot path.

        Parameters
        ----------
        event:
            High-priority event to insert at the front.
        """
        with self._lock:
            # Drain existing items
            buffer: list[BaseEvent] = []
            while not self._queue.empty():
                try:
                    buffer.append(self._queue.get_nowait())
                except queue.Empty:
                    break
            # Insert priority event first
            self._queue.put_nowait(event)
            # Re-queue drained items
            for buffered_event in buffer:
                self._queue.put_nowait(buffered_event)
            self._total_enqueued += 1

    def get(self, block: bool = False, timeout: Optional[float] = None) -> Optional[BaseEvent]:
        """Dequeue the next event.

        Parameters
        ----------
        block:
            If ``True``, wait until an event is available.
        timeout:
            Maximum seconds to wait when *block* is ``True``.  ``None``
            means wait indefinitely.

        Returns
        -------
        BaseEvent | None
            The next event, or ``None`` if the queue is empty and *block*
            is ``False`` (or the timeout expired).
        """
        try:
            event = self._queue.get(block=block, timeout=timeout)
            with self._lock:
                self._total_dequeued += 1
            return event
        except queue.Empty:
            return None

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def empty(self) -> bool:
        """Return ``True`` if the queue currently has no events."""
        return self._queue.empty()

    def qsize(self) -> int:
        """Return the *approximate* number of events in the queue.

        .. note::
            This value is only approximate due to concurrency; use it for
            monitoring and diagnostics, not for flow control.
        """
        return self._queue.qsize()

    @property
    def stats(self) -> dict[str, int]:
        """Return a snapshot of queue throughput statistics.

        Returns
        -------
        dict
            ``total_enqueued``, ``total_dequeued``, ``current_depth``.
        """
        with self._lock:
            return {
                "total_enqueued": self._total_enqueued,
                "total_dequeued": self._total_dequeued,
                "current_depth": self._total_enqueued - self._total_dequeued,
            }

    def flush(self) -> int:
        """Drain all remaining events without processing them.

        Returns
        -------
        int
            Number of events discarded.
        """
        discarded = 0
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                discarded += 1
            except queue.Empty:
                break
        return discarded

    def __len__(self) -> int:
        return self.qsize()

    def __repr__(self) -> str:
        return (
            f"EventQueue(depth={self.qsize()}, "
            f"enqueued={self._total_enqueued}, "
            f"dequeued={self._total_dequeued})"
        )
