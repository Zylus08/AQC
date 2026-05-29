"""
aqc/backtester/event.py
=======================
Core event dataclasses for the AQC event-driven backtesting engine.

Events flow through the system in the following order:

    MarketEvent → SignalEvent → OrderEvent → FillEvent → Portfolio Update

All events are immutable dataclasses.  Consumers should never mutate an event;
instead they emit a new downstream event.

Author: AQC Team
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class EventType(Enum):
    """Top-level event type discriminator."""

    MARKET = auto()
    SIGNAL = auto()
    ORDER = auto()
    FILL = auto()


class SignalDirection(Enum):
    """Directional intent of a trading signal."""

    LONG = "LONG"
    SHORT = "SHORT"
    EXIT = "EXIT"
    HOLD = "HOLD"


class OrderType(Enum):
    """Supported order types.

    Notes
    -----
    LIMIT and STOP_LIMIT are reserved for future execution engine
    implementations and are not yet processed by the default broker.
    """

    MARKET = "MARKET"
    LIMIT = "LIMIT"          # reserved
    STOP = "STOP"            # reserved
    STOP_LIMIT = "STOP_LIMIT"  # reserved


class OrderSide(Enum):
    """Side of an order."""

    BUY = "BUY"
    SELL = "SELL"


class FillStatus(Enum):
    """Terminal status of a fill."""

    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    REJECTED = "REJECTED"


# ---------------------------------------------------------------------------
# Event base
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BaseEvent:
    """Abstract base for all framework events.

    Attributes
    ----------
    event_type:
        Discriminator used by the event loop to route to the correct handler.
        Child classes set this automatically in ``__post_init__``; callers
        should **not** pass it explicitly.
    event_id:
        Universally unique identifier auto-generated at creation time.
    timestamp:
        Wall-clock time at which the event was created (UTC).
    """

    # Provide a sentinel default so subclasses can be constructed without
    # passing event_type.  __post_init__ in each subclass overwrites this.
    event_type: Optional[EventType] = field(default=None)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# MarketEvent
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MarketEvent(BaseEvent):
    """Emitted whenever a new bar of market data becomes available.

    The engine emits one ``MarketEvent`` per bar per symbol.  Strategies
    subscribe to these events and derive signals from the bar data.

    Attributes
    ----------
    symbol:
        Ticker / instrument identifier (e.g. ``"AAPL"``, ``"BTCUSDT"``).
    bar_time:
        Timestamp of the bar's open (index of the OHLCV row).
    open_price:
        Bar open price.
    high_price:
        Bar high price.
    low_price:
        Bar low price.
    close_price:
        Bar close price.
    volume:
        Bar volume (units traded).
    vwap:
        Volume-weighted average price for the bar (optional; ``None`` if not
        available in the source data).
    """

    symbol: str = ""
    bar_time: datetime = field(default_factory=datetime.utcnow)
    open_price: float = 0.0
    high_price: float = 0.0
    low_price: float = 0.0
    close_price: float = 0.0
    volume: float = 0.0
    vwap: Optional[float] = None

    def __post_init__(self) -> None:
        # Frozen dataclasses require object.__setattr__ for validation side effects
        object.__setattr__(self, "event_type", EventType.MARKET)


# ---------------------------------------------------------------------------
# SignalEvent
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SignalEvent(BaseEvent):
    """Emitted by a strategy to communicate a directional view.

    The portfolio engine translates signals into concrete orders after
    applying risk checks.

    Attributes
    ----------
    symbol:
        Target instrument.
    strategy_id:
        Human-readable identifier of the originating strategy (used for
        performance attribution).
    direction:
        Long, short, exit, or hold.
    strength:
        Normalised signal strength in ``[-1.0, 1.0]``.  A strength of ``1.0``
        means maximum conviction long; ``-1.0`` means maximum conviction short.
        The portfolio uses this to size positions proportionally (optional).
    suggested_price:
        Reference price at the time of signal generation.  Used for slippage
        and latency modelling.
    metadata:
        Arbitrary key/value pairs for debugging and research logging.
    """

    symbol: str = ""
    strategy_id: str = "unknown"
    direction: SignalDirection = SignalDirection.HOLD
    strength: float = 1.0
    suggested_price: Optional[float] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_type", EventType.SIGNAL)
        if not -1.0 <= self.strength <= 1.0:
            raise ValueError(
                f"Signal strength must be in [-1.0, 1.0], got {self.strength}"
            )


# ---------------------------------------------------------------------------
# OrderEvent
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OrderEvent(BaseEvent):
    """Emitted by the portfolio/risk layer; consumed by the execution engine.

    Attributes
    ----------
    symbol:
        Target instrument.
    order_type:
        Market, limit, stop, etc.
    side:
        Buy or sell.
    quantity:
        Number of units (shares, contracts, coins) to trade.  Always positive.
    limit_price:
        Required when ``order_type`` is ``LIMIT`` or ``STOP_LIMIT``.
    stop_price:
        Required when ``order_type`` is ``STOP`` or ``STOP_LIMIT``.
    strategy_id:
        Originating strategy identifier for PnL attribution.
    signal_ref:
        ``event_id`` of the ``SignalEvent`` that triggered this order (audit
        trail).
    """

    symbol: str = ""
    order_type: OrderType = OrderType.MARKET
    side: OrderSide = OrderSide.BUY
    quantity: float = 0.0
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    strategy_id: str = "unknown"
    signal_ref: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_type", EventType.ORDER)
        if self.quantity <= 0:
            raise ValueError(
                f"Order quantity must be positive, got {self.quantity}"
            )
        if self.order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT) and self.limit_price is None:
            raise ValueError("limit_price is required for LIMIT / STOP_LIMIT orders")
        if self.order_type in (OrderType.STOP, OrderType.STOP_LIMIT) and self.stop_price is None:
            raise ValueError("stop_price is required for STOP / STOP_LIMIT orders")


# ---------------------------------------------------------------------------
# FillEvent
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FillEvent(BaseEvent):
    """Emitted by the execution engine after an order is processed.

    Attributes
    ----------
    symbol:
        Filled instrument.
    exchange:
        Exchange or venue where the fill occurred.
    side:
        Buy or sell side of the fill.
    quantity:
        Units actually filled (may be less than ordered for partial fills).
    fill_price:
        Average execution price (includes slippage applied by the broker).
    commission:
        Total brokerage commission charged for this fill.
    status:
        ``FILLED``, ``PARTIALLY_FILLED``, or ``REJECTED``.
    order_ref:
        ``event_id`` of the originating ``OrderEvent`` (audit trail).
    strategy_id:
        Strategy identifier propagated from the order.
    """

    symbol: str = ""
    exchange: str = "SIMULATED"
    side: OrderSide = OrderSide.BUY
    quantity: float = 0.0
    fill_price: float = 0.0
    commission: float = 0.0
    status: FillStatus = FillStatus.FILLED
    order_ref: Optional[str] = None
    strategy_id: str = "unknown"

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_type", EventType.FILL)

    @property
    def gross_value(self) -> float:
        """Notional value of the fill before commission."""
        return self.quantity * self.fill_price

    @property
    def net_value(self) -> float:
        """Notional value including commission (signed: positive = cash out for BUY)."""
        sign = 1.0 if self.side == OrderSide.BUY else -1.0
        return sign * self.gross_value + self.commission
