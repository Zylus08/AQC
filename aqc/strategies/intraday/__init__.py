"""
aqc/strategies/intraday/__init__.py
====================================
Intraday Mean Reversion Strategy Suite.

This package contains four research-grade intraday mean reversion strategies
that integrate directly with the AQC ``BacktestEngine``:

* :class:`VWAPReversionStrategy` — Deviation from VWAP as a z-score signal.
* :class:`VolumeExhaustionStrategy` — Volume spike + failed breakout detector.
* :class:`ZScoreReversionStrategy` — Rolling z-score with adaptive thresholds.
* :class:`CompositeMeanReversionStrategy` — Multi-signal alpha composite.

All strategies inherit from :class:`~aqc.strategies.base_strategy.BaseStrategy`
and communicate via the standard event queue.

Author: Saksham Mishra — AlgoQuant Club
"""

from aqc.strategies.intraday.vwap_reversion import VWAPReversionStrategy
from aqc.strategies.intraday.volume_exhaustion import VolumeExhaustionStrategy
from aqc.strategies.intraday.zscore_reversion import ZScoreReversionStrategy
from aqc.strategies.intraday.composite_mean_reversion import CompositeMeanReversionStrategy

__all__ = [
    "VWAPReversionStrategy",
    "VolumeExhaustionStrategy",
    "ZScoreReversionStrategy",
    "CompositeMeanReversionStrategy",
]
