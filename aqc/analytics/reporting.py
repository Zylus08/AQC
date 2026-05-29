"""
aqc/analytics/reporting.py
==========================
Reporting utilities — console output and optional CSV export.

The :class:`ReportGenerator` formats backtest results into a human-readable
summary.  It is called automatically by :class:`~aqc.backtester.engine.BacktestEngine`
at the end of every run.

Author: AQC Team
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate formatted backtest reports.

    Parameters
    ----------
    portfolio_summary:
        Dictionary returned by :meth:`~aqc.backtester.portfolio.Portfolio.summary`.
    performance_metrics:
        Dictionary returned by :meth:`~aqc.analytics.metrics.PerformanceMetrics.compute_all`.
    equity_curve:
        Equity curve DataFrame.
    trade_log:
        List of trade dictionaries.

    Examples
    --------
    >>> reporter = ReportGenerator(portfolio_summary=ps, performance_metrics=pm, ...)
    >>> reporter.print_report()
    >>> reporter.save_report("reports/backtest_001.txt")
    """

    def __init__(
        self,
        portfolio_summary: dict,
        performance_metrics: dict,
        equity_curve: pd.DataFrame,
        trade_log: list[dict],
    ) -> None:
        self.portfolio_summary = portfolio_summary
        self.performance_metrics = performance_metrics
        self.equity_curve = equity_curve
        self.trade_log = trade_log

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def print_report(self) -> None:
        """Print the full summary report to stdout and the log."""
        report = self.build_report()
        print(report)
        logger.info("\n%s", report)

    def save_report(self, filepath: str) -> None:
        """Save the report to a text file.

        Parameters
        ----------
        filepath:
            Output path (parent directories will be created if needed).
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        report = self.build_report()
        path.write_text(report, encoding="utf-8")
        logger.info("Report saved to %s", path)

    def export_equity_curve(self, filepath: str) -> None:
        """Export the equity curve to a CSV file.

        Parameters
        ----------
        filepath:
            Output CSV path.
        """
        if self.equity_curve.empty:
            logger.warning("Equity curve is empty — nothing to export.")
            return
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.equity_curve.to_csv(path)
        logger.info("Equity curve exported to %s", path)

    def export_trade_log(self, filepath: str) -> None:
        """Export the trade log to a CSV file.

        Parameters
        ----------
        filepath:
            Output CSV path.
        """
        if not self.trade_log:
            logger.warning("Trade log is empty — nothing to export.")
            return
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.trade_log).to_csv(path, index=False)
        logger.info("Trade log exported to %s", path)

    def build_report(self) -> str:
        """Build the full report as a string.

        Returns
        -------
        str
            Formatted report text.
        """
        sep = "=" * 65
        thin_sep = "-" * 65

        lines: list[str] = [
            "",
            sep,
            "  AQC BACKTEST REPORT",
            sep,
            "",
            "  PORTFOLIO SUMMARY",
            thin_sep,
        ]

        ps = self.portfolio_summary
        lines += [
            f"  Initial Capital    : {self._fmt_currency(ps.get('initial_capital', 0))}",
            f"  Final Equity       : {self._fmt_currency(ps.get('final_equity', 0))}",
            f"  Total PnL          : {self._fmt_pnl(ps.get('total_pnl', 0))}",
            f"  Realised PnL       : {self._fmt_pnl(ps.get('total_realised_pnl', 0))}",
            f"  Unrealised PnL     : {self._fmt_pnl(ps.get('total_unrealised_pnl', 0))}",
            f"  Total Commission   : {self._fmt_currency(ps.get('total_commission', 0))}",
            f"  Total Return       : {ps.get('return_pct', 0):+.2f}%",
            f"  Num Trades         : {ps.get('num_trades', 0)}",
            f"  Open Positions     : {ps.get('num_open_positions', 0)}",
            "",
            "  PERFORMANCE METRICS",
            thin_sep,
        ]

        pm = self.performance_metrics
        if pm:
            lines += [
                f"  CAGR               : {self._fmt_pct(pm.get('cagr', 0))}",
                f"  Sharpe Ratio       : {self._fmt_float(pm.get('sharpe_ratio'))}",
                f"  Sortino Ratio      : {self._fmt_float(pm.get('sortino_ratio'))}",
                f"  Calmar Ratio       : {self._fmt_float(pm.get('calmar_ratio'))}",
                f"  Max Drawdown       : {self._fmt_float(pm.get('max_drawdown_pct'))}%",
                f"  Max Drawdown (abs) : {self._fmt_currency(pm.get('max_drawdown_abs', 0))}",
                f"  Ann. Volatility    : {self._fmt_pct(pm.get('annualised_volatility', 0))}",
                f"  Win Rate           : {self._fmt_pct(pm.get('win_rate', 0))}",
                f"  Profit Factor      : {self._fmt_float(pm.get('profit_factor'))}",
                f"  Avg Trade Return   : {self._fmt_currency(pm.get('avg_trade_return', 0))}",
                f"  Avg Win            : {self._fmt_currency(pm.get('avg_win', 0))}",
                f"  Avg Loss           : {self._fmt_currency(pm.get('avg_loss', 0))}",
                f"  Total Return       : {self._fmt_pct(pm.get('total_return_pct', 0) / 100)}",
                f"  Exposure           : {self._fmt_pct(pm.get('exposure', 0))}",
                f"  Wins / Losses      : {pm.get('num_wins', 0)} / {pm.get('num_losses', 0)}",
            ]
        else:
            lines.append("  [No performance metrics computed]")

        lines += ["", sep, ""]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fmt_currency(value: Optional[float]) -> str:
        if value is None or (isinstance(value, float) and not _is_finite(value)):
            return "N/A"
        return f"${value:>12,.2f}"

    @staticmethod
    def _fmt_pnl(value: Optional[float]) -> str:
        if value is None or (isinstance(value, float) and not _is_finite(value)):
            return "N/A"
        sign = "+" if value >= 0 else ""
        return f"{sign}${value:>11,.2f}"

    @staticmethod
    def _fmt_pct(value: Optional[float]) -> str:
        if value is None or (isinstance(value, float) and not _is_finite(value)):
            return "N/A"
        return f"{value * 100:>+.2f}%"

    @staticmethod
    def _fmt_float(value: Optional[float]) -> str:
        if value is None or (isinstance(value, float) and not _is_finite(value)):
            return "N/A"
        return f"{value:>+.4f}"


def _is_finite(value: float) -> bool:
    """Return True if value is a finite number."""
    import math
    return math.isfinite(value)
