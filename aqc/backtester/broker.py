"""
aqc/backtester/broker.py
========================
Simulated broker and commission / slippage models.

The broker receives :class:`~aqc.backtester.event.OrderEvent` objects and
converts them to :class:`~aqc.backtester.event.FillEvent` objects, applying
configurable commission and slippage models.

Design
------
* :class:`CommissionModel` and :class:`SlippageModel` are abstract base
  classes.  Concrete implementations can be swapped via dependency injection
  without touching the broker or engine.
* :class:`SimulatedBroker` is the default execution handler for backtesting.
  It assumes fills are always possible at the bar's close price (market
  orders only).
* The interface is designed so that a ``LiveBroker`` implementation can
  accept the same ``OrderEvent`` and emit the same ``FillEvent`` — enabling
  live trading with minimal code changes.

Author: AQC Team
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

from aqc.backtester.event import (
    FillEvent,
    FillStatus,
    MarketEvent,
    OrderEvent,
    OrderSide,
    OrderType,
)

if TYPE_CHECKING:
    from aqc.backtester.event_queue import EventQueue

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Commission models
# ---------------------------------------------------------------------------


class CommissionModel(ABC):
    """Abstract base class for commission calculations.

    All implementations must override :meth:`calculate`.
    """

    @abstractmethod
    def calculate(self, order: OrderEvent, fill_price: float) -> float:
        """Return the commission charged for a given fill.

        Parameters
        ----------
        order:
            The order being filled.
        fill_price:
            Execution price (after slippage).

        Returns
        -------
        float
            Commission amount in the account currency.
        """


class PercentageCommission(CommissionModel):
    """Charges a fixed percentage of notional value.

    Parameters
    ----------
    rate:
        Commission rate (e.g. ``0.001`` = 0.1%).
    minimum:
        Minimum commission per trade (floor).

    Examples
    --------
    >>> cm = PercentageCommission(rate=0.001, minimum=1.0)
    >>> cm.calculate(order, 100.0)  # 100 shares @ 100 → 10 000 notional → 10.0 commission
    """

    def __init__(self, rate: float = 0.001, minimum: float = 0.0) -> None:
        self.rate = rate
        self.minimum = minimum

    def calculate(self, order: OrderEvent, fill_price: float) -> float:
        notional = order.quantity * fill_price
        return max(self.minimum, notional * self.rate)


class FlatFeeCommission(CommissionModel):
    """Charges a fixed fee per trade regardless of size.

    Parameters
    ----------
    fee:
        Commission per trade in account currency.
    """

    def __init__(self, fee: float = 5.0) -> None:
        self.fee = fee

    def calculate(self, order: OrderEvent, fill_price: float) -> float:
        return self.fee


class ZeroCommission(CommissionModel):
    """No commission — useful for crypto or zero-fee brokers."""

    def calculate(self, order: OrderEvent, fill_price: float) -> float:
        return 0.0


# ---------------------------------------------------------------------------
# Slippage models
# ---------------------------------------------------------------------------


class SlippageModel(ABC):
    """Abstract base class for slippage calculations.

    All implementations must override :meth:`apply`.
    """

    @abstractmethod
    def apply(self, order: OrderEvent, market_price: float) -> float:
        """Return the final fill price after applying slippage.

        Parameters
        ----------
        order:
            The order being filled.
        market_price:
            Baseline execution price (e.g. close of the bar).

        Returns
        -------
        float
            Adjusted fill price with slippage applied.
        """


class FixedBpsSlippage(SlippageModel):
    """Applies a fixed number of basis points of slippage.

    Slippage is always *adverse* — buys fill higher, sells fill lower.

    Parameters
    ----------
    bps:
        Basis points of slippage (1 bps = 0.01%).

    Examples
    --------
    >>> sm = FixedBpsSlippage(bps=5)
    >>> sm.apply(buy_order, 100.0)   # → 100.05
    >>> sm.apply(sell_order, 100.0)  # → 99.95
    """

    def __init__(self, bps: float = 5.0) -> None:
        self.bps = bps

    def apply(self, order: OrderEvent, market_price: float) -> float:
        factor = self.bps / 10_000.0
        if order.side == OrderSide.BUY:
            return market_price * (1.0 + factor)
        return market_price * (1.0 - factor)


class ZeroSlippage(SlippageModel):
    """No slippage — fills at exactly the market price."""

    def apply(self, order: OrderEvent, market_price: float) -> float:
        return market_price


# ---------------------------------------------------------------------------
# Execution handler protocol
# ---------------------------------------------------------------------------


class ExecutionHandler(ABC):
    """Interface for execution handlers (broker adapters).

    Concrete implementations include :class:`SimulatedBroker` (backtesting)
    and future ``LiveBroker`` (live trading).
    """

    @abstractmethod
    def execute_order(
        self,
        order: OrderEvent,
        market_event: Optional[MarketEvent] = None,
    ) -> Optional[FillEvent]:
        """Attempt to fill *order* and return a :class:`~aqc.backtester.event.FillEvent`.

        Parameters
        ----------
        order:
            The order to execute.
        market_event:
            The most recent bar for the instrument (provides the reference
            fill price for simulated brokers).

        Returns
        -------
        FillEvent | None
            A filled or rejected event, or ``None`` if the order cannot be
            processed (e.g. stale market data).
        """


# ---------------------------------------------------------------------------
# Simulated broker
# ---------------------------------------------------------------------------


class SimulatedBroker(ExecutionHandler):
    """Simulated broker for backtesting.

    Fills :class:`~aqc.backtester.event.OrderEvent` objects at the close
    price of the current bar, adjusted for slippage and commission.

    Only :attr:`~aqc.backtester.event.OrderType.MARKET` orders are supported.
    LIMIT/STOP orders will be rejected with a warning.

    Parameters
    ----------
    event_queue:
        Shared event queue.  Filled :class:`~aqc.backtester.event.FillEvent`
        objects are placed here.
    commission_model:
        Commission calculation strategy.  Defaults to
        :class:`PercentageCommission` at 0.1%.
    slippage_model:
        Slippage calculation strategy.  Defaults to :class:`FixedBpsSlippage`
        at 5 bps.
    exchange:
        Label to tag fills with (used for analytics).

    Examples
    --------
    >>> broker = SimulatedBroker(event_queue=eq)
    >>> fill = broker.execute_order(order, market_event)
    """

    def __init__(
        self,
        event_queue: "EventQueue",
        commission_model: Optional[CommissionModel] = None,
        slippage_model: Optional[SlippageModel] = None,
        exchange: str = "SIMULATED",
    ) -> None:
        self._eq = event_queue
        self._commission = commission_model or PercentageCommission(rate=0.001)
        self._slippage = slippage_model or FixedBpsSlippage(bps=5)
        self.exchange = exchange

    def execute_order(
        self,
        order: OrderEvent,
        market_event: Optional[MarketEvent] = None,
    ) -> Optional[FillEvent]:
        """Fill a MARKET order at the bar close with slippage and commission.

        Parameters
        ----------
        order:
            Incoming order to fill.
        market_event:
            Most recent bar for the instrument.

        Returns
        -------
        FillEvent | None
            A ``FILLED`` event, or ``None`` if market data is unavailable.
        """
        if order.order_type != OrderType.MARKET:
            logger.warning(
                "SimulatedBroker only supports MARKET orders. "
                "Received %s — skipping.",
                order.order_type.value,
            )
            return None

        if market_event is None:
            logger.error(
                "Cannot fill order for %s — no market data available.",
                order.symbol,
            )
            return None

        # Reference price: close of the current bar
        market_price = market_event.close_price
        fill_price = self._slippage.apply(order, market_price)
        commission = self._commission.calculate(order, fill_price)

        fill = FillEvent(
            symbol=order.symbol,
            exchange=self.exchange,
            side=order.side,
            quantity=order.quantity,
            fill_price=fill_price,
            commission=commission,
            status=FillStatus.FILLED,
            order_ref=order.event_id,
            strategy_id=order.strategy_id,
        )

        logger.info(
            "Fill generated: %s %s %.2f @ %.4f  slippage=%.4f  commission=%.4f",
            fill.side.value,
            fill.symbol,
            fill.quantity,
            fill.fill_price,
            fill_price - market_price,
            commission,
        )

        self._eq.put(fill)
        return fill

    def __repr__(self) -> str:
        return (
            f"SimulatedBroker("
            f"commission={self._commission.__class__.__name__}, "
            f"slippage={self._slippage.__class__.__name__})"
        )
