"""
aqc/diagnostics/trade_reports.py
==================================
CSV & console report generation for trade-level analysis.

Author: Saksham Mishra — AlgoQuant Club
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class TradeReportGenerator:
    """Generate formatted trade-level reports.

    Parameters
    ----------
    trades_df : pd.DataFrame
        From ``TradeAnalyzer.to_dataframe()``.
    attribution : dict[str, pd.DataFrame]
        From ``TradeAttributionEngine.full_summary()``.
    trade_stats : dict
        From ``TradeAnalyzer.trade_stats()``.
    """

    def __init__(
        self,
        trades_df: pd.DataFrame,
        attribution: Optional[dict[str, pd.DataFrame]] = None,
        trade_stats: Optional[dict] = None,
    ) -> None:
        self.trades_df = trades_df
        self.attribution = attribution or {}
        self.trade_stats = trade_stats or {}

    def print_report(self) -> None:
        """Print trade attribution report to console."""
        report = self.build_report()
        print(report)

    def build_report(self) -> str:
        sep = "=" * 70
        thin = "-" * 70
        lines = ["", sep, "  AQC TRADE ATTRIBUTION REPORT", sep, ""]

        # Trade stats
        if self.trade_stats:
            lines += ["  TRADE STATISTICS", thin]
            for k, v in self.trade_stats.items():
                lines.append(f"  {k:24s} : {v}")
            lines.append("")

        # Top 5 winners
        if not self.trades_df.empty:
            lines += ["  TOP 5 WINNING TRADES", thin]
            top_w = self.trades_df.nlargest(5, "realised_pnl")
            for _, t in top_w.iterrows():
                lines.append(
                    f"  {t.get('symbol',''):6s}  {t.get('direction',''):5s}  "
                    f"PnL=${t.get('realised_pnl', 0):>10,.2f}  "
                    f"Ret={t.get('realised_return', 0)*100:>+6.2f}%  "
                    f"Dur={t.get('holding_duration_days', 0):.1f}d  "
                    f"Regime={t.get('vol_regime', '')}"
                )
            lines.append("")

            # Top 5 losers
            lines += ["  TOP 5 LOSING TRADES", thin]
            top_l = self.trades_df.nsmallest(5, "realised_pnl")
            for _, t in top_l.iterrows():
                lines.append(
                    f"  {t.get('symbol',''):6s}  {t.get('direction',''):5s}  "
                    f"PnL=${t.get('realised_pnl', 0):>10,.2f}  "
                    f"Ret={t.get('realised_return', 0)*100:>+6.2f}%  "
                    f"Dur={t.get('holding_duration_days', 0):.1f}d  "
                    f"Regime={t.get('vol_regime', '')}"
                )
            lines.append("")

        # Attribution tables
        for name, df in self.attribution.items():
            if df.empty or name.startswith("heatmap"):
                continue
            lines += [f"  PnL BY {name.upper().replace('BY_', '')}", thin]
            for _, row in df.iterrows():
                lines.append(
                    f"  {str(row.get('label', ''))[:18]:18s}  "
                    f"N={int(row.get('n_trades', 0)):>4d}  "
                    f"PnL=${float(row.get('total_pnl', 0)):>10,.2f}  "
                    f"WR={float(row.get('win_rate', 0))*100:.1f}%  "
                    f"Contrib={float(row.get('pct_of_total_pnl', 0)):>+6.1f}%"
                )
            lines.append("")

        lines += [sep, ""]
        return "\n".join(lines)

    def save_all(self, output_dir: str = "reports") -> None:
        """Save all CSV reports."""
        p = Path(output_dir)
        p.mkdir(parents=True, exist_ok=True)

        if not self.trades_df.empty:
            self.trades_df.to_csv(p / "trade_attribution_report.csv", index=False)
            self.trades_df.nlargest(10, "realised_pnl").to_csv(
                p / "top_winners.csv", index=False
            )
            self.trades_df.nsmallest(10, "realised_pnl").to_csv(
                p / "top_losers.csv", index=False
            )

        for name, df in self.attribution.items():
            if not df.empty:
                df.to_csv(p / f"{name}.csv", index=True)

        logger.info("Trade reports saved to %s", output_dir)
