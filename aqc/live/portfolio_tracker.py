"""
aqc/live/portfolio_tracker.py
===============================
Real-time wrapper for Portfolio to support snapshotting and exposure tracking.

Author: Saksham Mishra — AlgoQuant Club
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from aqc.backtester.portfolio import Portfolio

logger = logging.getLogger(__name__)


@dataclass
class PortfolioSnapshot:
    timestamp: pd.Timestamp
    total_equity: float
    cash: float
    gross_exposure: float
    net_exposure: float
    num_positions: int
    unrealised_pnl: float
    realised_pnl: float
    leverage: float


class PortfolioTracker:
    """Tracks and snapshots portfolio state in real-time.

    Parameters
    ----------
    portfolio : Portfolio
        The underlying accounting engine.
    """

    def __init__(self, portfolio: "Portfolio") -> None:
        self.portfolio = portfolio
        self.snapshots: list[PortfolioSnapshot] = []

    def snapshot(self) -> PortfolioSnapshot:
        """Create a point-in-time snapshot of portfolio metrics."""
        equity = self.portfolio.equity
        cash = self.portfolio.cash
        pos = self.portfolio.positions
        
        long_exp = sum(p.quantity * p.last_price for p in pos.values() if p.quantity > 0)
        short_exp = sum(abs(p.quantity) * p.last_price for p in pos.values() if p.quantity < 0)
        
        gross_exp = long_exp + short_exp
        net_exp = long_exp - short_exp
        
        unrealised = sum(p.unrealised_pnl for p in pos.values())
        realised = sum(p.realised_pnl for p in pos.values())
        
        leverage = gross_exp / equity if equity > 0 else 0.0

        snap = PortfolioSnapshot(
            timestamp=pd.Timestamp.utcnow(),
            total_equity=equity,
            cash=cash,
            gross_exposure=gross_exp,
            net_exposure=net_exp,
            num_positions=len(pos),
            unrealised_pnl=unrealised,
            realised_pnl=realised,
            leverage=leverage,
        )
        
        self.snapshots.append(snap)
        return snap

    def to_dataframe(self) -> pd.DataFrame:
        if not self.snapshots:
            return pd.DataFrame()
        return pd.DataFrame([s.__dict__ for s in self.snapshots]).set_index("timestamp")
