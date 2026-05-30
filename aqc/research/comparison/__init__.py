"""
aqc/research/comparison/__init__.py
=====================================
Comparative Backtesting Framework.

Author: Saksham Mishra — AlgoQuant Club
"""

from aqc.research.comparison.comparator import BacktestComparator, StatisticalTests
from aqc.research.comparison.reporting import ComparisonReportGenerator

__all__ = [
    "BacktestComparator",
    "StatisticalTests",
    "ComparisonReportGenerator",
]
