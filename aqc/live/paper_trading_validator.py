"""
aqc/live/paper_trading_validator.py
=====================================
Validates paper trading performance against backtest performance.

Author: Saksham Mishra — AlgoQuant Club
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from aqc.analytics.metrics import PerformanceMetrics

logger = logging.getLogger(__name__)


class PaperTradingValidator:
    """Compares live/paper execution against an equivalent backtest."""

    def __init__(self, backtest_equity: pd.Series, paper_equity: pd.Series) -> None:
        self.bt_equity = backtest_equity
        self.paper_equity = paper_equity

    def validate(self) -> pd.DataFrame:
        """Generate a comparison report of key metrics."""
        # Align timeframes if possible, or just compare directly if same dates
        
        bt_pm = PerformanceMetrics(self.bt_equity.to_frame(name="equity"), [])
        bt_metrics = bt_pm.compute_all()
        
        paper_pm = PerformanceMetrics(self.paper_equity.to_frame(name="equity"), [])
        paper_metrics = paper_pm.compute_all()
        
        # We can also compare execution costs if provided, but for now we focus on equity curves
        
        data = {
            "Metric": ["CAGR", "Sharpe", "Max Drawdown", "Final Equity"],
            "Backtest": [
                bt_metrics.get("cagr", 0.0),
                bt_metrics.get("sharpe_ratio", 0.0),
                bt_metrics.get("max_drawdown_pct", 0.0),
                self.bt_equity.iloc[-1] if not self.bt_equity.empty else 0
            ],
            "Paper": [
                paper_metrics.get("cagr", 0.0),
                paper_metrics.get("sharpe_ratio", 0.0),
                paper_metrics.get("max_drawdown_pct", 0.0),
                self.paper_equity.iloc[-1] if not self.paper_equity.empty else 0
            ]
        }
        
        df = pd.DataFrame(data)
        df["Difference"] = df["Paper"] - df["Backtest"]
        return df

    def save_report(self, df: pd.DataFrame, output_dir: str = "reports") -> None:
        p = Path(output_dir)
        p.mkdir(parents=True, exist_ok=True)
        df.to_csv(p / "validation_report.csv", index=False)
        logger.info("Validation report saved to %s", p / "validation_report.csv")
