"""
aqc/strategies/intraday/vwap_reversion.py
==========================================
VWAP Deviation Mean Reversion Strategy.

Alpha thesis
------------
Intraday prices tend to revert toward the Volume-Weighted Average Price
(VWAP).  When price deviates significantly from VWAP — measured as a
z-score of ``(price - VWAP) / rolling_std`` — there is a statistical
tendency for price to revert.

The strategy:

1. Computes a cumulative VWAP from the bar data.
2. Measures the z-score of the deviation: ``z = (close - VWAP) / std``.
3. Enters **long** when ``z < -entry_threshold`` (price below VWAP).
4. Enters **short** when ``z > +entry_threshold`` (price above VWAP).
5. Exits when ``|z| < exit_threshold`` (price reverts to VWAP).
6. Applies a time-based exit after ``max_holding_bars`` bars.
7. Enforces stop-loss at ``stop_loss_pct`` from entry.

Signal strength is proportional to ``|z| / entry_threshold``, clamped
to ``[-1, 1]``.

Parameters
----------
entry_threshold : float
    Z-score threshold to trigger entry (default 2.0).
exit_threshold : float
    Z-score threshold to trigger exit (default 0.5).
rolling_window : int
    Window for rolling std calculation (default 20).
max_holding_bars : int
    Maximum bars to hold a position before forced exit (default 20).
stop_loss_pct : float
    Stop loss as a fraction of entry price (default 0.02 = 2%).

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from aqc.backtester.event import SignalDirection, SignalEvent
from aqc.strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class VWAPReversionStrategy(BaseStrategy):
    """VWAP deviation mean reversion strategy.

    Trades the z-score of ``(price - VWAP) / rolling_std`` with
    configurable entry/exit thresholds, holding period limits, and
    stop-loss protection.

    Parameters
    ----------
    event_queue:
        Shared event queue.
    symbols:
        Instruments to trade.
    entry_threshold:
        Z-score magnitude to enter (default 2.0).
    exit_threshold:
        Z-score magnitude to exit (default 0.5).
    rolling_window:
        Bars for rolling std (default 20).
    max_holding_bars:
        Force-exit after this many bars (default 20, 0 = disabled).
    stop_loss_pct:
        Stop-loss fraction from entry price (default 0.02).
    strategy_id:
        Strategy label.

    Examples
    --------
    >>> strategy = VWAPReversionStrategy(
    ...     event_queue=eq, symbols=["AAPL"],
    ...     entry_threshold=2.0, exit_threshold=0.5,
    ... )
    """

    def __init__(
        self,
        event_queue,
        symbols: list[str],
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
        rolling_window: int = 20,
        max_holding_bars: int = 20,
        stop_loss_pct: float = 0.02,
        strategy_id: str = "vwap_reversion",
    ) -> None:
        super().__init__(
            event_queue=event_queue,
            symbols=symbols,
            strategy_id=strategy_id,
            lookback=max(rolling_window * 3, 100),
        )
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.rolling_window = rolling_window
        self.max_holding_bars = max_holding_bars
        self.stop_loss_pct = stop_loss_pct

        # Per-symbol state
        self._position: dict[str, Optional[SignalDirection]] = {s: None for s in symbols}
        self._entry_price: dict[str, float] = {s: 0.0 for s in symbols}
        self._bars_held: dict[str, int] = {s: 0 for s in symbols}

    @property
    def min_bars_required(self) -> int:
        return self.rolling_window + 5

    # ------------------------------------------------------------------
    # VWAP calculation
    # ------------------------------------------------------------------

    @staticmethod
    def compute_vwap(bars: pd.DataFrame) -> pd.Series:
        """Compute cumulative VWAP from OHLCV bars.

        Parameters
        ----------
        bars:
            DataFrame with ``close``, ``high``, ``low``, ``volume`` columns.

        Returns
        -------
        pd.Series
            Cumulative VWAP.
        """
        typical_price = (bars["high"] + bars["low"] + bars["close"]) / 3.0
        cum_tp_vol = (typical_price * bars["volume"]).cumsum()
        cum_vol = bars["volume"].cumsum().replace(0, np.nan)
        return cum_tp_vol / cum_vol

    @staticmethod
    def compute_vwap_zscore(
        close: pd.Series, vwap: pd.Series, window: int
    ) -> pd.Series:
        """Compute the z-score of price deviation from VWAP.

        Parameters
        ----------
        close:
            Close price series.
        vwap:
            VWAP series.
        window:
            Rolling window for standard deviation.

        Returns
        -------
        pd.Series
            Z-score series.
        """
        deviation = close - vwap
        rolling_std = deviation.rolling(window=window, min_periods=window).std()
        rolling_std = rolling_std.replace(0, np.nan)
        return deviation / rolling_std

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate_signal(self, symbol: str, bars: pd.DataFrame) -> Optional[SignalEvent]:
        """Generate VWAP deviation mean-reversion signals.

        Parameters
        ----------
        symbol:
            Instrument to check.
        bars:
            Rolling bar window.

        Returns
        -------
        SignalEvent | None
        """
        close = bars["close"]
        vwap = self.compute_vwap(bars)
        z = self.compute_vwap_zscore(close, vwap, self.rolling_window)

        if z.isna().iloc[-1]:
            return None

        current_z = float(z.iloc[-1])
        current_price = float(close.iloc[-1])
        current_pos = self._position[symbol]

        # --- Stop-loss / time-based exit ---
        if current_pos is not None:
            self._bars_held[symbol] += 1

            # Time exit
            if self.max_holding_bars > 0 and self._bars_held[symbol] >= self.max_holding_bars:
                return self._exit(symbol, current_price, reason="time_exit", z=current_z)

            # Stop-loss
            entry = self._entry_price[symbol]
            if current_pos == SignalDirection.LONG:
                if current_price <= entry * (1.0 - self.stop_loss_pct):
                    return self._exit(symbol, current_price, reason="stop_loss", z=current_z)
            elif current_pos == SignalDirection.SHORT:
                if current_price >= entry * (1.0 + self.stop_loss_pct):
                    return self._exit(symbol, current_price, reason="stop_loss", z=current_z)

        # --- Mean-reversion exit: z crosses back to neutral ---
        if current_pos == SignalDirection.LONG and current_z >= -self.exit_threshold:
            return self._exit(symbol, current_price, reason="z_revert", z=current_z)
        if current_pos == SignalDirection.SHORT and current_z <= self.exit_threshold:
            return self._exit(symbol, current_price, reason="z_revert", z=current_z)

        # --- Entry signals ---
        if current_pos is None:
            # Long: price significantly below VWAP
            if current_z < -self.entry_threshold:
                strength = min(1.0, abs(current_z) / (self.entry_threshold * 2))
                self._position[symbol] = SignalDirection.LONG
                self._entry_price[symbol] = current_price
                self._bars_held[symbol] = 0
                return SignalEvent(
                    symbol=symbol,
                    strategy_id=self.strategy_id,
                    direction=SignalDirection.LONG,
                    strength=strength,
                    suggested_price=current_price,
                    metadata={
                        "vwap": round(float(vwap.iloc[-1]), 4),
                        "z_score": round(current_z, 4),
                        "signal_type": "vwap_reversion",
                    },
                )
            # Short: price significantly above VWAP
            elif current_z > self.entry_threshold:
                strength = min(1.0, abs(current_z) / (self.entry_threshold * 2))
                self._position[symbol] = SignalDirection.SHORT
                self._entry_price[symbol] = current_price
                self._bars_held[symbol] = 0
                return SignalEvent(
                    symbol=symbol,
                    strategy_id=self.strategy_id,
                    direction=SignalDirection.SHORT,
                    strength=strength,
                    suggested_price=current_price,
                    metadata={
                        "vwap": round(float(vwap.iloc[-1]), 4),
                        "z_score": round(current_z, 4),
                        "signal_type": "vwap_reversion",
                    },
                )

        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _exit(
        self, symbol: str, price: float, reason: str, z: float
    ) -> SignalEvent:
        """Build an exit signal and reset position state.

        Parameters
        ----------
        symbol:
            Instrument.
        price:
            Current close price.
        reason:
            Exit reason tag (for metadata).
        z:
            Current z-score.

        Returns
        -------
        SignalEvent
        """
        self._position[symbol] = None
        self._entry_price[symbol] = 0.0
        self._bars_held[symbol] = 0
        return SignalEvent(
            symbol=symbol,
            strategy_id=self.strategy_id,
            direction=SignalDirection.EXIT,
            strength=1.0,
            suggested_price=price,
            metadata={
                "exit_reason": reason,
                "z_score": round(z, 4),
                "signal_type": "vwap_reversion",
            },
        )
