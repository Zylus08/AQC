"""
aqc/live/performance_tracker.py
=================================
Calculates rolling performance metrics for the live trading dashboard.

Author: Saksham Mishra — AlgoQuant Club
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from aqc.analytics.metrics import PerformanceMetrics

logger = logging.getLogger(__name__)


@dataclass
class LivePerformanceMetrics:
    timestamp: pd.Timestamp
    daily_return: float
    cagr: float
    sharpe: float
    sortino: float
    max_drawdown: float
    win_rate: float
    profit_factor: float


class PerformanceTracker:
    """Computes live rolling performance metrics.

    Parameters
    ----------
    equity_curve : pd.Series
        Time series of total equity.
    trade_log : list[dict]
        Completed trades for win rate / profit factor.
    """

    def __init__(self) -> None:
        self.history: list[LivePerformanceMetrics] = []

    def compute_metrics(
        self,
        equity_series: pd.Series,
        trade_df: Optional[pd.DataFrame] = None
    ) -> LivePerformanceMetrics:
        """Compute current metrics."""
        
        # Guard
        if len(equity_series) < 2:
            return LivePerformanceMetrics(pd.Timestamp.utcnow(), 0,0,0,0,0,0,0)

        eq_df = equity_series.to_frame(name="equity")
        trades = trade_df.to_dict('records') if trade_df is not None and not trade_df.empty else []
        pm = PerformanceMetrics(eq_df, trades)
        metrics = pm.compute_all()

        win_rate = metrics.get("win_rate", 0.0)
        profit_factor = metrics.get("profit_factor", 0.0)

        live_metrics = LivePerformanceMetrics(
            timestamp=pd.Timestamp.utcnow(),
            daily_return=metrics.get("cagr", 0.0), # CAGR is in the dictionary
            cagr=metrics.get("cagr", 0.0),
            sharpe=metrics.get("sharpe_ratio", 0.0),
            sortino=metrics.get("sortino_ratio", 0.0),
            max_drawdown=metrics.get("max_drawdown_pct", 0.0),
            win_rate=win_rate,
            profit_factor=profit_factor
        )
        
        self.history.append(live_metrics)
        return live_metrics
