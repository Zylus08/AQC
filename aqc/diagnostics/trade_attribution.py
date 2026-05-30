"""
aqc/diagnostics/trade_attribution.py
=======================================
PnL decomposition engine — attribute portfolio returns to regime, duration,
signal type, and symbol.

Author: Saksham Mishra — AlgoQuant Club
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class AttributionBucket:
    """PnL attribution for a single category."""
    category: str = ""
    label: str = ""
    n_trades: int = 0
    total_pnl: float = 0.0
    avg_pnl: float = 0.0
    win_rate: float = 0.0
    avg_return: float = 0.0
    avg_duration: float = 0.0
    sharpe_contribution: float = 0.0
    pct_of_total_pnl: float = 0.0


class TradeAttributionEngine:
    """Decompose PnL by multiple dimensions.

    Parameters
    ----------
    trades_df : pd.DataFrame
        Output of ``TradeAnalyzer.to_dataframe()``.
    """

    def __init__(self, trades_df: pd.DataFrame) -> None:
        self.df = trades_df.copy() if not trades_df.empty else pd.DataFrame()

    # ------------------------------------------------------------------
    # PnL by Vol Regime
    # ------------------------------------------------------------------

    def by_vol_regime(self) -> pd.DataFrame:
        """PnL broken down by volatility regime."""
        return self._decompose("vol_regime")

    # ------------------------------------------------------------------
    # PnL by Trend Regime
    # ------------------------------------------------------------------

    def by_trend_regime(self) -> pd.DataFrame:
        """PnL broken down by trend regime."""
        return self._decompose("trend_regime")

    # ------------------------------------------------------------------
    # PnL by Duration Bucket
    # ------------------------------------------------------------------

    def by_duration(self) -> pd.DataFrame:
        """PnL broken down by holding duration bucket."""
        return self._decompose("holding_duration_bucket")

    # ------------------------------------------------------------------
    # PnL by Signal Source
    # ------------------------------------------------------------------

    def by_signal(self) -> pd.DataFrame:
        """PnL broken down by signal source / strategy_id."""
        return self._decompose("signal_source")

    # ------------------------------------------------------------------
    # PnL by Symbol
    # ------------------------------------------------------------------

    def by_symbol(self) -> pd.DataFrame:
        """PnL broken down by symbol."""
        return self._decompose("symbol")

    # ------------------------------------------------------------------
    # PnL by Direction
    # ------------------------------------------------------------------

    def by_direction(self) -> pd.DataFrame:
        """PnL broken down by trade direction (LONG/SHORT)."""
        return self._decompose("direction")

    # ------------------------------------------------------------------
    # Cross-attribution: Regime × Duration
    # ------------------------------------------------------------------

    def heatmap_regime_duration(self) -> pd.DataFrame:
        """PnL heatmap: vol_regime × holding_duration_bucket."""
        if self.df.empty:
            return pd.DataFrame()
        return self.df.pivot_table(
            values="realised_pnl",
            index="vol_regime",
            columns="holding_duration_bucket",
            aggfunc="sum",
            fill_value=0,
        )

    def heatmap_regime_signal(self) -> pd.DataFrame:
        """PnL heatmap: vol_regime × signal_source."""
        if self.df.empty:
            return pd.DataFrame()
        return self.df.pivot_table(
            values="realised_pnl",
            index="vol_regime",
            columns="signal_source",
            aggfunc="sum",
            fill_value=0,
        )

    def heatmap_trend_duration(self) -> pd.DataFrame:
        """PnL heatmap: trend_regime × holding_duration_bucket."""
        if self.df.empty:
            return pd.DataFrame()
        return self.df.pivot_table(
            values="realised_pnl",
            index="trend_regime",
            columns="holding_duration_bucket",
            aggfunc="sum",
            fill_value=0,
        )

    # ------------------------------------------------------------------
    # Full decomposition summary
    # ------------------------------------------------------------------

    def full_summary(self) -> dict[str, pd.DataFrame]:
        """Run all decompositions and return a dict of DataFrames."""
        return {
            "by_vol_regime": self.by_vol_regime(),
            "by_trend_regime": self.by_trend_regime(),
            "by_duration": self.by_duration(),
            "by_signal": self.by_signal(),
            "by_symbol": self.by_symbol(),
            "by_direction": self.by_direction(),
            "heatmap_regime_duration": self.heatmap_regime_duration(),
            "heatmap_regime_signal": self.heatmap_regime_signal(),
            "heatmap_trend_duration": self.heatmap_trend_duration(),
        }

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _decompose(self, column: str) -> pd.DataFrame:
        """Generic decomposition by a categorical column."""
        if self.df.empty or column not in self.df.columns:
            return pd.DataFrame()

        total_pnl = float(self.df["realised_pnl"].sum())
        total_abs = abs(total_pnl) if abs(total_pnl) > 1e-10 else 1.0

        rows = []
        for label, grp in self.df.groupby(column, dropna=False):
            if len(grp) == 0:
                continue
            pnl = float(grp["realised_pnl"].sum())
            wins = grp[grp["realised_pnl"] > 0]
            ret_col = "realised_return" if "realised_return" in grp.columns else "realised_pnl"
            dur_col = "holding_duration_days" if "holding_duration_days" in grp.columns else None

            rows.append({
                "category": column,
                "label": str(label),
                "n_trades": len(grp),
                "total_pnl": round(pnl, 2),
                "avg_pnl": round(float(grp["realised_pnl"].mean()), 2),
                "win_rate": round(len(wins) / len(grp), 4),
                "avg_return": round(float(grp[ret_col].mean()), 6),
                "avg_duration": round(float(grp[dur_col].mean()), 2) if dur_col else 0,
                "pct_of_total_pnl": round(pnl / total_abs * 100, 2),
            })

        return pd.DataFrame(rows)
