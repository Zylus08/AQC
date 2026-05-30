"""
aqc/diagnostics/__init__.py
============================
Portfolio Diagnostics & Validation Framework.

Forensic analysis layer that explains WHY the portfolio behaved the way it did.

Modules:
* LeverageAnalyzer — gross/net leverage tracking and regime decomposition
* ExposureAnalyzer — long/short/gross/net exposure forensics
* RiskBudgetAnalyzer — vol-target utilisation and risk budget compliance
* PositionAnalyzer — position size, concentration, and turnover analysis
* RegimePerformanceAnalyzer — per-regime performance breakdown
* ForecastAnalyzer — volatility forecast accuracy validation
* PerformanceAttributionEngine — return decomposition by source
* DiagnosticsEngine — composite orchestrator
* DiagnosticsReportGenerator — CSV + plot generation
* PortfolioDiagnosticsDashboard — HTML dashboard
* PortfolioValidator — automated health checks with scoring

Author: Saksham Mishra — AlgoQuant Club
"""

from aqc.diagnostics.leverage_analysis import LeverageAnalyzer
from aqc.diagnostics.exposure_analysis import ExposureAnalyzer
from aqc.diagnostics.risk_budget_analysis import RiskBudgetAnalyzer
from aqc.diagnostics.position_analysis import PositionAnalyzer
from aqc.diagnostics.regime_analysis import RegimePerformanceAnalyzer
from aqc.diagnostics.forecast_analysis import ForecastAnalyzer
from aqc.diagnostics.attribution import PerformanceAttributionEngine
from aqc.diagnostics.diagnostics_engine import DiagnosticsEngine, PortfolioValidator

# Trade-Level Attribution
from aqc.diagnostics.trade_analyzer import TradeAnalyzer, TradeRecord
from aqc.diagnostics.trade_attribution import TradeAttributionEngine
from aqc.diagnostics.trade_reports import TradeReportGenerator
from aqc.diagnostics.trade_visualization import TradeVisualizer

__all__ = [
    "LeverageAnalyzer",
    "ExposureAnalyzer",
    "RiskBudgetAnalyzer",
    "PositionAnalyzer",
    "RegimePerformanceAnalyzer",
    "ForecastAnalyzer",
    "PerformanceAttributionEngine",
    "DiagnosticsEngine",
    "PortfolioValidator",
    "TradeAnalyzer",
    "TradeRecord",
    "TradeAttributionEngine",
    "TradeReportGenerator",
    "TradeVisualizer",
]
