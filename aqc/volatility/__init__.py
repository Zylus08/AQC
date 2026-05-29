"""
aqc/volatility/__init__.py
===========================
Volatility Forecasting Framework.

Provides production-grade volatility estimation and forecasting:

* :func:`ewma_volatility` — Exponentially Weighted Moving Average vol.
* :func:`realized_volatility` — Rolling realized (historical) vol.
* :class:`GARCH11` — GARCH(1,1) maximum-likelihood estimator.
* :class:`VolatilityForecastEngine` — Multi-model forecasting orchestrator.
* :class:`VolatilitySizer` — Vol-targeting position sizer.

Author: Saksham Mishra — AlgoQuant Club
"""

from aqc.volatility.ewma import ewma_volatility, ewma_variance
from aqc.volatility.garch import GARCH11
from aqc.volatility.forecasting_engine import (
    VolatilityForecastEngine,
    VolRegime,
    ForecastResult,
)
from aqc.volatility.volatility_metrics import (
    VolatilitySizer,
    SizingMethod,
    volatility_cone,
    vol_of_vol,
    forecast_error_stats,
)

__all__ = [
    "ewma_volatility",
    "ewma_variance",
    "GARCH11",
    "VolatilityForecastEngine",
    "VolRegime",
    "ForecastResult",
    "VolatilitySizer",
    "SizingMethod",
    "volatility_cone",
    "vol_of_vol",
    "forecast_error_stats",
]
