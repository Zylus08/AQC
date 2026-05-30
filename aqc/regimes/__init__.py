"""
aqc/regimes/__init__.py
========================
Market Regime Detection Framework.

Provides multi-dimensional market state classification:

* :class:`VolatilityRegimeDetector` — LOW / NORMAL / HIGH / EXTREME
* :class:`TrendRegimeDetector` — 5-state trend classification
* :class:`CorrelationRegimeDetector` — Cross-asset correlation regimes
* :class:`HMMRegimeDetector` — Gaussian Hidden Markov Model states
* :class:`RegimeEngine` — Composite orchestrator
* :class:`RegimeFilter` — Strategy/regime permission gate

Author: Saksham Mishra — AlgoQuant Club
"""

from aqc.regimes.volatility_regime import VolatilityRegimeDetector, VolatilityRegime
from aqc.regimes.trend_regime import TrendRegimeDetector, TrendRegime
from aqc.regimes.correlation_regime import CorrelationRegimeDetector, CorrelationRegime
from aqc.regimes.hmm_regime import HMMRegimeDetector, HMMState
from aqc.regimes.regime_engine import (
    RegimeEngine,
    RegimeSnapshot,
    RegimeFilter,
)

__all__ = [
    "VolatilityRegimeDetector",
    "VolatilityRegime",
    "TrendRegimeDetector",
    "TrendRegime",
    "CorrelationRegimeDetector",
    "CorrelationRegime",
    "HMMRegimeDetector",
    "HMMState",
    "RegimeEngine",
    "RegimeSnapshot",
    "RegimeFilter",
]
