"""
aqc/portfolio/volatility_portfolio.py
=======================================
Volatility-Targeted Portfolio.

Extends the core ``Portfolio`` with volatility-forecast-aware position sizing
and optional regime filtering.

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

from aqc.backtester.event import SignalEvent, SignalDirection
from aqc.backtester.portfolio import Portfolio

if TYPE_CHECKING:
    from aqc.backtester.event_queue import EventQueue
    from aqc.regimes.regime_engine import RegimeEngine, RegimeFilter, RegimeSnapshot
    from aqc.risk.risk_manager import RiskManager
    from aqc.volatility.forecasting_engine import VolatilityForecastEngine
    from aqc.volatility.volatility_metrics import VolatilitySizer

logger = logging.getLogger(__name__)


class VolatilityTargetedPortfolio(Portfolio):
    """Portfolio with volatility-targeted position sizing.

    Overrides ``_size_order()`` to use volatility forecasts for sizing
    and optionally consults a ``RegimeFilter`` before placing trades.

    Parameters
    ----------
    event_queue:
        Shared event queue.
    risk_manager:
        Risk gate.
    vol_engine:
        Volatility forecasting engine (provides per-symbol forecasts).
    vol_sizer:
        Volatility-based position sizer.
    regime_engine:
        Optional regime detector.
    regime_filter:
        Optional regime filter (gates signals by regime).
    strategy_type:
        Strategy archetype for regime filtering (e.g., "mean_reversion").
    price_history:
        Dict mapping symbol → list of close prices (updated by on_market_event).
    initial_capital:
        Starting cash.
    default_quantity:
        Fallback quantity when vol forecast is unavailable.

    Examples
    --------
    >>> from aqc.volatility import VolatilityForecastEngine, VolatilitySizer
    >>> vtport = VolatilityTargetedPortfolio(
    ...     event_queue=eq, risk_manager=rm,
    ...     vol_engine=VolatilityForecastEngine(),
    ...     vol_sizer=VolatilitySizer(target_vol=0.10),
    ... )
    """

    def __init__(
        self,
        event_queue: "EventQueue",
        risk_manager: "RiskManager",
        vol_engine: "VolatilityForecastEngine",
        vol_sizer: "VolatilitySizer",
        regime_engine: Optional["RegimeEngine"] = None,
        regime_filter: Optional["RegimeFilter"] = None,
        strategy_type: str = "mean_reversion",
        initial_capital: float = 100_000.0,
        default_quantity: float = 100.0,
        min_history: int = 50,
    ) -> None:
        super().__init__(
            event_queue=event_queue,
            risk_manager=risk_manager,
            initial_capital=initial_capital,
            default_quantity=default_quantity,
        )
        self.vol_engine = vol_engine
        self.vol_sizer = vol_sizer
        self.regime_engine = regime_engine
        self.regime_filter = regime_filter
        self.strategy_type = strategy_type
        self.min_history = min_history

        # Per-symbol price tracking
        self._price_history: dict[str, list[float]] = defaultdict(list)
        self._ohlc_history: dict[str, list[dict]] = defaultdict(list)

        # Logging
        self.vol_targeting_log: list[dict] = []
        self._last_regime: Optional["RegimeSnapshot"] = None
        self._regime_blocked_count: int = 0

    # ------------------------------------------------------------------
    # Override: market event tracking
    # ------------------------------------------------------------------

    def on_market_event(self, event) -> None:
        """Track price history and delegate to parent."""
        self._price_history[event.symbol].append(event.close_price)
        self._ohlc_history[event.symbol].append({
            "open": event.open_price,
            "high": event.high_price,
            "low": event.low_price,
            "close": event.close_price,
            "volume": event.volume,
        })
        super().on_market_event(event)

    # ------------------------------------------------------------------
    # Override: signal filtering (regime gate)
    # ------------------------------------------------------------------

    def on_signal_event(self, event: SignalEvent) -> None:
        """Apply regime filter before delegating to parent."""
        if self.regime_engine is not None and self.regime_filter is not None:
            prices = self._get_price_series(event.symbol)
            if prices is not None and len(prices) >= self.min_history:
                ohlc_df = self._get_ohlc_df(event.symbol)
                snapshot = self.regime_engine.detect(prices, ohlc_df=ohlc_df)
                self._last_regime = snapshot

                if not self.regime_filter.should_trade(self.strategy_type, snapshot):
                    self._regime_blocked_count += 1
                    logger.debug(
                        "Signal blocked by regime filter: %s vol=%s trend=%s",
                        event.symbol,
                        snapshot.volatility_regime.value,
                        snapshot.trend_regime.value,
                    )
                    return

        super().on_signal_event(event)

    # ------------------------------------------------------------------
    # Override: vol-targeted sizing
    # ------------------------------------------------------------------

    def _size_order(self, signal: SignalEvent) -> float:
        """Compute position size using volatility forecast.

        Falls back to the parent's proportional sizing if vol forecast
        is unavailable (insufficient price history).
        """
        prices = self._get_price_series(signal.symbol)

        if prices is None or len(prices) < self.min_history:
            qty = super()._size_order(signal)
            self._log_sizing(signal, qty, forecast_vol=None, method="fixed_fallback")
            return qty

        # Get vol forecast
        forecast = self.vol_engine.fit_and_forecast(prices)

        if forecast.forecast_vol <= 0:
            qty = super()._size_order(signal)
            self._log_sizing(signal, qty, forecast_vol=0.0, method="fixed_fallback")
            return qty

        # Vol-targeted sizing
        price = float(prices.iloc[-1])
        result = self.vol_sizer.size_position(
            symbol=signal.symbol,
            price=price,
            forecast_vol=forecast.forecast_vol,
            equity=self.equity,
        )

        qty = max(1.0, float(result.quantity)) * abs(signal.strength)
        qty = round(qty, 4)

        self._log_sizing(
            signal, qty,
            forecast_vol=forecast.forecast_vol,
            method="vol_target",
            regime=self._last_regime,
        )

        return qty

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_price_series(self, symbol: str) -> Optional[pd.Series]:
        """Build a price Series from tracked history."""
        history = self._price_history.get(symbol)
        if not history or len(history) < 2:
            return None
        return pd.Series(history, name="close")

    def _get_ohlc_df(self, symbol: str) -> Optional[pd.DataFrame]:
        """Build an OHLC DataFrame from tracked history."""
        history = self._ohlc_history.get(symbol)
        if not history or len(history) < 2:
            return None
        return pd.DataFrame(history)

    def _log_sizing(
        self,
        signal: SignalEvent,
        quantity: float,
        forecast_vol: Optional[float],
        method: str,
        regime: Optional["RegimeSnapshot"] = None,
    ) -> None:
        """Record vol-targeting metadata for post-analysis."""
        entry = {
            "symbol": signal.symbol,
            "direction": signal.direction.value,
            "strength": signal.strength,
            "quantity": quantity,
            "forecast_vol": forecast_vol,
            "method": method,
            "equity": round(self.equity, 2),
        }
        if regime is not None:
            entry["vol_regime"] = regime.volatility_regime.value
            entry["trend_regime"] = regime.trend_regime.value
        self.vol_targeting_log.append(entry)

    def vol_targeting_summary(self) -> dict:
        """Return summary statistics of vol-targeting behaviour."""
        log = self.vol_targeting_log
        if not log:
            return {}

        vol_targeted = [e for e in log if e["method"] == "vol_target"]
        fallback = [e for e in log if e["method"] == "fixed_fallback"]

        vols = [e["forecast_vol"] for e in vol_targeted if e["forecast_vol"] is not None]

        return {
            "total_signals": len(log),
            "vol_targeted_count": len(vol_targeted),
            "fallback_count": len(fallback),
            "regime_blocked_count": self._regime_blocked_count,
            "avg_forecast_vol": round(float(np.mean(vols)), 6) if vols else None,
            "min_forecast_vol": round(float(np.min(vols)), 6) if vols else None,
            "max_forecast_vol": round(float(np.max(vols)), 6) if vols else None,
        }
