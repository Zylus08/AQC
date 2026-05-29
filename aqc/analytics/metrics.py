"""
aqc/analytics/metrics.py
=========================
Performance metrics for post-backtest analysis.

The :class:`PerformanceMetrics` class takes the equity curve and trade log
produced by the portfolio and computes a comprehensive set of risk-adjusted
return metrics.

Metrics implemented
-------------------
* Sharpe Ratio (annualised)
* Sortino Ratio (annualised)
* Maximum Drawdown (absolute and percentage)
* Calmar Ratio
* CAGR (Compound Annual Growth Rate)
* Win Rate
* Profit Factor
* Average Trade Return
* Average Win / Average Loss
* Exposure (fraction of bars with an open position)
* Volatility (annualised return std)

Author: AQC Team
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE_ANNUAL = 0.04  # 4% — override in config


class PerformanceMetrics:
    """Compute risk-adjusted performance metrics from backtest results.

    Parameters
    ----------
    equity_curve:
        DataFrame with a :class:`~pandas.DatetimeIndex` and an ``equity``
        column produced by the portfolio.
    trade_log:
        List of trade dictionaries (from ``portfolio.trade_log``).
    risk_free_rate:
        Annual risk-free rate used for Sharpe / Sortino calculation.
        Defaults to ``RISK_FREE_RATE_ANNUAL`` (4%).
    periods_per_year:
        Number of bars per year (252 for daily, 52 for weekly, etc.).

    Examples
    --------
    >>> metrics = PerformanceMetrics(equity_curve=eq_df, trade_log=trades)
    >>> results = metrics.compute_all()
    """

    def __init__(
        self,
        equity_curve: pd.DataFrame,
        trade_log: list[dict],
        risk_free_rate: float = RISK_FREE_RATE_ANNUAL,
        periods_per_year: int = TRADING_DAYS_PER_YEAR,
    ) -> None:
        self.equity_curve = equity_curve
        self.trade_log = trade_log
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year

        # Pre-compute daily returns from the equity curve
        self._returns: pd.Series = self._compute_returns()
        self._trade_df: pd.DataFrame = self._build_trade_df()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def compute_all(self) -> dict:
        """Compute every available metric and return as a dictionary.

        Returns
        -------
        dict
            All metrics keyed by name.
        """
        if self.equity_curve.empty or len(self._returns) < 2:
            logger.warning("Insufficient equity curve data for metric computation.")
            return {}

        return {
            "sharpe_ratio": self.sharpe_ratio(),
            "sortino_ratio": self.sortino_ratio(),
            "max_drawdown_pct": self.max_drawdown_pct(),
            "max_drawdown_abs": self.max_drawdown_abs(),
            "cagr": self.cagr(),
            "calmar_ratio": self.calmar_ratio(),
            "annualised_volatility": self.annualised_volatility(),
            "win_rate": self.win_rate(),
            "profit_factor": self.profit_factor(),
            "avg_trade_return": self.avg_trade_return(),
            "avg_win": self.avg_win(),
            "avg_loss": self.avg_loss(),
            "num_trades": len(self._trade_df),
            "num_wins": int(self._trade_df["pnl"].gt(0).sum()) if not self._trade_df.empty else 0,
            "num_losses": int(self._trade_df["pnl"].lt(0).sum()) if not self._trade_df.empty else 0,
            "exposure": self.exposure(),
            "total_return_pct": self.total_return_pct(),
        }

    # ------------------------------------------------------------------
    # Return / Growth metrics
    # ------------------------------------------------------------------

    def total_return_pct(self) -> float:
        """Total return as a percentage.

        Returns
        -------
        float
            ``(final_equity / initial_equity - 1) * 100``
        """
        eq = self.equity_curve["equity"]
        if len(eq) < 2:
            return 0.0
        return round((eq.iloc[-1] / eq.iloc[0] - 1.0) * 100.0, 4)

    def cagr(self) -> float:
        """Compound Annual Growth Rate.

        Returns
        -------
        float
            CAGR as a decimal (e.g. ``0.15`` = 15%).
        """
        eq = self.equity_curve["equity"]
        if len(eq) < 2 or eq.iloc[0] <= 0:
            return 0.0
        n_years = len(eq) / self.periods_per_year
        if n_years <= 0:
            return 0.0
        return round((eq.iloc[-1] / eq.iloc[0]) ** (1.0 / n_years) - 1.0, 6)

    # ------------------------------------------------------------------
    # Risk-adjusted return metrics
    # ------------------------------------------------------------------

    def sharpe_ratio(self) -> float:
        """Annualised Sharpe Ratio.

        ``Sharpe = (E[R] - Rf) / σ(R) * sqrt(T)``

        Returns
        -------
        float
            Sharpe ratio (``NaN`` if standard deviation is zero).
        """
        if self._returns.empty:
            return float("nan")
        daily_rf = self.risk_free_rate / self.periods_per_year
        excess = self._returns - daily_rf
        std = excess.std()
        if std < 1e-10:
            return float("nan")
        return round(float(excess.mean() / std * math.sqrt(self.periods_per_year)), 4)

    def sortino_ratio(self) -> float:
        """Annualised Sortino Ratio.

        Uses downside deviation (std of negative returns only) in the
        denominator.

        Returns
        -------
        float
        """
        if self._returns.empty:
            return float("nan")
        daily_rf = self.risk_free_rate / self.periods_per_year
        excess = self._returns - daily_rf
        downside = excess[excess < 0]
        if len(downside) < 2:
            return float("nan")
        downside_std = downside.std()
        if downside_std < 1e-10:
            return float("nan")
        return round(float(excess.mean() / downside_std * math.sqrt(self.periods_per_year)), 4)

    def calmar_ratio(self) -> float:
        """Calmar Ratio = CAGR / |Max Drawdown|.

        Returns
        -------
        float
        """
        mdd = self.max_drawdown_pct()
        if mdd == 0.0:
            return float("nan")
        return round(self.cagr() / abs(mdd / 100.0), 4)

    # ------------------------------------------------------------------
    # Drawdown metrics
    # ------------------------------------------------------------------

    def max_drawdown_pct(self) -> float:
        """Maximum Drawdown as a percentage.

        Returns
        -------
        float
            Max drawdown (always non-positive, e.g. ``-15.0`` = -15%).
        """
        eq = self.equity_curve["equity"]
        if len(eq) < 2:
            return 0.0
        rolling_max = eq.cummax()
        drawdown = (eq - rolling_max) / rolling_max * 100.0
        return round(float(drawdown.min()), 4)

    def max_drawdown_abs(self) -> float:
        """Maximum Drawdown in absolute currency units.

        Returns
        -------
        float
        """
        eq = self.equity_curve["equity"]
        if len(eq) < 2:
            return 0.0
        rolling_max = eq.cummax()
        return round(float((eq - rolling_max).min()), 2)

    # ------------------------------------------------------------------
    # Volatility
    # ------------------------------------------------------------------

    def annualised_volatility(self) -> float:
        """Annualised return standard deviation.

        Returns
        -------
        float
        """
        if len(self._returns) < 2:
            return 0.0
        return round(float(self._returns.std() * math.sqrt(self.periods_per_year)), 6)

    # ------------------------------------------------------------------
    # Trade-level metrics
    # ------------------------------------------------------------------

    def win_rate(self) -> float:
        """Fraction of trades that were profitable.

        Returns
        -------
        float
            Win rate in ``[0, 1]`` (e.g. ``0.55`` = 55%).
        """
        if self._trade_df.empty:
            return float("nan")
        wins = (self._trade_df["pnl"] > 0).sum()
        return round(float(wins / len(self._trade_df)), 4)

    def profit_factor(self) -> float:
        """Ratio of gross profit to gross loss.

        ``Profit Factor = Σ(winning trades) / |Σ(losing trades)|``

        Returns
        -------
        float
            Values > 1 indicate a profitable system.
        """
        if self._trade_df.empty:
            return float("nan")
        gross_profit = self._trade_df[self._trade_df["pnl"] > 0]["pnl"].sum()
        gross_loss = abs(self._trade_df[self._trade_df["pnl"] < 0]["pnl"].sum())
        if gross_loss < 1e-10:
            return float("inf") if gross_profit > 0 else float("nan")
        return round(float(gross_profit / gross_loss), 4)

    def avg_trade_return(self) -> float:
        """Average return per trade.

        Returns
        -------
        float
        """
        if self._trade_df.empty:
            return float("nan")
        return round(float(self._trade_df["pnl"].mean()), 4)

    def avg_win(self) -> float:
        """Average profit on winning trades.

        Returns
        -------
        float
        """
        if self._trade_df.empty:
            return float("nan")
        wins = self._trade_df[self._trade_df["pnl"] > 0]["pnl"]
        return round(float(wins.mean()), 4) if not wins.empty else 0.0

    def avg_loss(self) -> float:
        """Average loss on losing trades (signed, so always ≤ 0).

        Returns
        -------
        float
        """
        if self._trade_df.empty:
            return float("nan")
        losses = self._trade_df[self._trade_df["pnl"] < 0]["pnl"]
        return round(float(losses.mean()), 4) if not losses.empty else 0.0

    def exposure(self) -> float:
        """Fraction of bars during which at least one position was open.

        Returns
        -------
        float
            Exposure in ``[0, 1]``.
        """
        if self.equity_curve.empty or "num_positions" not in self.equity_curve.columns:
            return float("nan")
        exposed = (self.equity_curve.get("num_positions", pd.Series(dtype=float)) > 0).sum()
        return round(float(exposed / len(self.equity_curve)), 4)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_returns(self) -> pd.Series:
        """Compute per-bar returns from the equity curve."""
        if self.equity_curve.empty or "equity" not in self.equity_curve.columns:
            return pd.Series(dtype=float)
        return self.equity_curve["equity"].pct_change().dropna()

    def _build_trade_df(self) -> pd.DataFrame:
        """Convert the raw trade log into a DataFrame with PnL per round trip."""
        if not self.trade_log:
            return pd.DataFrame(columns=["pnl"])
        df = pd.DataFrame(self.trade_log)
        df["pnl"] = df.get("realised_pnl", pd.Series(dtype=float)).fillna(0)
        return df[df["pnl"] != 0].reset_index(drop=True)

    def __repr__(self) -> str:
        return (
            f"PerformanceMetrics("
            f"bars={len(self.equity_curve)}, "
            f"trades={len(self._trade_df)})"
        )
