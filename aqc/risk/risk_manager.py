"""
aqc/risk/risk_manager.py
========================
Rule-based risk management layer.

The :class:`RiskManager` acts as a gating mechanism between the portfolio
and the broker.  Before any order is submitted, the portfolio calls
:meth:`validate_order` to check:

1. Maximum single-position size.
2. Maximum portfolio gross exposure.
3. Maximum daily loss.

If any rule is violated the order is rejected and the reason is returned
to the caller for logging.

The design is deliberately extensible:

* Override :meth:`_custom_rules` to add strategy-specific constraints.
* Future additions (VaR, volatility targeting, regime filters) slot in as
  additional private methods called from :meth:`validate_order`.

Author: AQC Team
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Optional

from aqc.backtester.event import FillEvent, OrderEvent, OrderSide

if TYPE_CHECKING:
    from aqc.backtester.portfolio import Portfolio

logger = logging.getLogger(__name__)


@dataclass
class RiskConfig:
    """Configuration dataclass for risk parameters.

    Attributes
    ----------
    max_position_size:
        Maximum number of units allowed in a single position.  Set to
        ``float('inf')`` to disable.
    max_position_pct_equity:
        Maximum single-position value as a fraction of total equity (0–1).
        E.g. ``0.20`` = 20% of account equity.
    max_gross_exposure_pct:
        Maximum gross exposure (sum of |positions|) as a fraction of equity.
        E.g. ``2.0`` = 200% gross exposure (allows leverage).
    max_daily_loss_pct:
        Maximum allowable drawdown from today's opening equity, expressed as
        a fraction.  E.g. ``0.02`` = 2% maximum intraday loss.
    max_open_positions:
        Maximum number of simultaneously open positions.
    """

    max_position_size: float = float("inf")
    max_position_pct_equity: float = 0.20
    max_gross_exposure_pct: float = 1.0
    max_daily_loss_pct: float = 0.05
    max_open_positions: int = 10


class RiskManager:
    """Validates orders against configurable risk limits.

    Parameters
    ----------
    config:
        :class:`RiskConfig` instance.  If ``None``, default limits apply.

    Examples
    --------
    >>> from aqc.risk.risk_manager import RiskManager, RiskConfig
    >>> risk = RiskManager(config=RiskConfig(max_daily_loss_pct=0.02))
    >>> approved, reason = risk.validate_order(order, portfolio)
    """

    def __init__(self, config: Optional[RiskConfig] = None) -> None:
        self.config = config or RiskConfig()
        self._daily_start_equity: dict[date, float] = {}
        self._daily_pnl: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate_order(
        self, order: OrderEvent, portfolio: "Portfolio"
    ) -> tuple[bool, str]:
        """Check an order against all active risk rules.

        Parameters
        ----------
        order:
            Order to validate.
        portfolio:
            Current portfolio state (provides equity, positions, etc.).

        Returns
        -------
        tuple[bool, str]
            ``(approved, reason)``

            * If approved: ``(True, "")``
            * If rejected: ``(False, "<human-readable reason>")``
        """
        checks = [
            self._check_position_size(order),
            self._check_position_pct_equity(order, portfolio),
            self._check_gross_exposure(order, portfolio),
            self._check_daily_loss(portfolio),
            self._check_max_positions(order, portfolio),
            self._custom_rules(order, portfolio),
        ]

        for approved, reason in checks:
            if not approved:
                logger.warning("Risk check FAILED for %s: %s", order.symbol, reason)
                return False, reason

        return True, ""

    def record_fill(self, fill: FillEvent) -> None:
        """Update daily PnL tracking after a fill.

        Called by the portfolio after every fill.  Tracks intraday PnL to
        enforce the daily loss limit.

        Parameters
        ----------
        fill:
            Confirmed fill event.
        """
        today = date.today()
        # The daily PnL impact of this fill is approximated by commission cost
        # (full realised PnL is tracked by the portfolio)
        self._daily_pnl -= fill.commission

    def reset_daily_state(self, equity: float) -> None:
        """Record today's opening equity for daily loss calculations.

        Call this at the start of each trading day (or at engine init).

        Parameters
        ----------
        equity:
            Current portfolio equity to use as the daily baseline.
        """
        today = date.today()
        self._daily_start_equity[today] = equity
        self._daily_pnl = 0.0
        logger.info("Daily risk state reset: start_equity=%.2f", equity)

    # ------------------------------------------------------------------
    # Individual rule checks
    # ------------------------------------------------------------------

    def _check_position_size(self, order: OrderEvent) -> tuple[bool, str]:
        """Enforce maximum position size limit."""
        if order.quantity > self.config.max_position_size:
            return (
                False,
                f"Position size {order.quantity} exceeds max "
                f"{self.config.max_position_size}",
            )
        return True, ""

    def _check_position_pct_equity(
        self, order: OrderEvent, portfolio: "Portfolio"
    ) -> tuple[bool, str]:
        """Reject if the order's notional value exceeds the equity % limit."""
        equity = portfolio.equity
        if equity <= 0:
            return True, ""  # avoid division by zero edge case

        # Use the last known price from the current position, or a rough estimate
        pos = portfolio.positions.get(order.symbol)
        price = pos.last_price if pos and pos.last_price > 0 else 1.0
        notional = order.quantity * price

        if notional / equity > self.config.max_position_pct_equity:
            return (
                False,
                f"Order notional {notional:.2f} ({notional/equity:.1%} of equity) "
                f"exceeds limit of {self.config.max_position_pct_equity:.1%}",
            )
        return True, ""

    def _check_gross_exposure(
        self, order: OrderEvent, portfolio: "Portfolio"
    ) -> tuple[bool, str]:
        """Reject if gross exposure after the order would exceed the limit."""
        equity = portfolio.equity
        if equity <= 0:
            return True, ""

        pos = portfolio.positions.get(order.symbol)
        price = pos.last_price if pos and pos.last_price > 0 else 1.0
        additional_exposure = order.quantity * price

        projected_exposure = portfolio.gross_exposure + additional_exposure
        limit = equity * self.config.max_gross_exposure_pct

        if projected_exposure > limit:
            return (
                False,
                f"Projected gross exposure {projected_exposure:.2f} "
                f"({projected_exposure/equity:.1%}) exceeds limit "
                f"{self.config.max_gross_exposure_pct:.1%}",
            )
        return True, ""

    def _check_daily_loss(self, portfolio: "Portfolio") -> tuple[bool, str]:
        """Halt trading if daily loss limit is breached."""
        today = date.today()
        start_equity = self._daily_start_equity.get(today)

        if start_equity is None:
            # First order of the day — set baseline
            self._daily_start_equity[today] = portfolio.equity
            return True, ""

        current_equity = portfolio.equity
        daily_loss = (start_equity - current_equity) / start_equity

        if daily_loss > self.config.max_daily_loss_pct:
            return (
                False,
                f"Daily loss {daily_loss:.2%} exceeds max allowed "
                f"{self.config.max_daily_loss_pct:.2%}",
            )
        return True, ""

    def _check_max_positions(
        self, order: OrderEvent, portfolio: "Portfolio"
    ) -> tuple[bool, str]:
        """Reject new position opens if max open positions is reached."""
        if order.symbol in portfolio.positions and not portfolio.positions[order.symbol].is_flat:
            # Modifying an existing position — always allowed
            return True, ""

        if portfolio.num_open_positions >= self.config.max_open_positions:
            return (
                False,
                f"Max open positions ({self.config.max_open_positions}) reached",
            )
        return True, ""

    def _custom_rules(
        self, order: OrderEvent, portfolio: "Portfolio"
    ) -> tuple[bool, str]:
        """Hook for subclass-defined custom risk rules.

        Override this in subclasses to add strategy-specific constraints
        without modifying the core :class:`RiskManager`.

        Parameters
        ----------
        order:
            Order to validate.
        portfolio:
            Current portfolio state.

        Returns
        -------
        tuple[bool, str]
            By default always returns ``(True, "")``.
        """
        return True, ""

    def __repr__(self) -> str:
        return (
            f"RiskManager("
            f"max_pos_size={self.config.max_position_size}, "
            f"max_pos_pct={self.config.max_position_pct_equity:.0%}, "
            f"max_exposure={self.config.max_gross_exposure_pct:.0%}, "
            f"max_daily_loss={self.config.max_daily_loss_pct:.0%})"
        )
