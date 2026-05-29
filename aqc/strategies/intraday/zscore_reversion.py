"""
aqc/strategies/intraday/zscore_reversion.py
============================================
Rolling Z-Score Mean Reversion Strategy with Adaptive Thresholds.

Alpha thesis
------------
When a price's rolling z-score exceeds a threshold derived from the
recent volatility regime, the move is statistically over-extended and
likely to revert.  This strategy adapts its entry thresholds to the
current regime:

* **Low-volatility regime** — tighter thresholds (smaller deviations
  are significant).
* **High-volatility regime** — wider thresholds (need a larger deviation
  to be meaningful).

The z-score is computed as:

    z = (close - rolling_mean) / rolling_std

Adaptive threshold is:

    threshold = base_threshold * (1 + vol_adjustment * vol_ratio)

where ``vol_ratio = current_vol / long_term_vol`` and ``vol_adjustment``
is a configurable scaling factor (default 0.5).

Parameters
----------
z_window : int
    Rolling window for mean/std calculation (default 20).
base_entry_z : float
    Base z-score for entry in neutral vol regime (default 2.0).
base_exit_z : float
    Z-score threshold for exit (default 0.5).
vol_lookback : int
    Longer window for vol regime estimation (default 60).
vol_adjustment : float
    How much to scale thresholds based on vol ratio (default 0.5).
max_holding_bars : int
    Force-exit after this many bars (default 15).
stop_loss_pct : float
    Stop-loss fraction from entry (default 0.02).
allow_short : bool
    If True, short when z > +threshold (default True).

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


class ZScoreReversionStrategy(BaseStrategy):
    """Rolling z-score mean reversion with adaptive vol-based thresholds.

    Parameters
    ----------
    event_queue:
        Shared event queue.
    symbols:
        Instruments to trade.
    z_window:
        Rolling window for z-score (default 20).
    base_entry_z:
        Base entry z-score threshold (default 2.0).
    base_exit_z:
        Exit z-score threshold (default 0.5).
    vol_lookback:
        Longer window for volatility regime (default 60).
    vol_adjustment:
        Threshold scaling factor (default 0.5).
    max_holding_bars:
        Force exit after this many bars (default 15, 0 = disabled).
    stop_loss_pct:
        Stop-loss fraction from entry (default 0.02).
    allow_short:
        Enable short signals (default True).
    strategy_id:
        Strategy label.

    Examples
    --------
    >>> strategy = ZScoreReversionStrategy(
    ...     event_queue=eq, symbols=["AAPL"],
    ...     z_window=20, base_entry_z=2.0,
    ... )
    """

    def __init__(
        self,
        event_queue,
        symbols: list[str],
        z_window: int = 20,
        base_entry_z: float = 2.0,
        base_exit_z: float = 0.5,
        vol_lookback: int = 60,
        vol_adjustment: float = 0.5,
        max_holding_bars: int = 15,
        stop_loss_pct: float = 0.02,
        allow_short: bool = True,
        strategy_id: str = "zscore_reversion",
    ) -> None:
        super().__init__(
            event_queue=event_queue,
            symbols=symbols,
            strategy_id=strategy_id,
            lookback=max(vol_lookback * 2, z_window * 4),
        )
        self.z_window = z_window
        self.base_entry_z = base_entry_z
        self.base_exit_z = base_exit_z
        self.vol_lookback = vol_lookback
        self.vol_adjustment = vol_adjustment
        self.max_holding_bars = max_holding_bars
        self.stop_loss_pct = stop_loss_pct
        self.allow_short = allow_short

        self._position: dict[str, Optional[SignalDirection]] = {s: None for s in symbols}
        self._entry_price: dict[str, float] = {s: 0.0 for s in symbols}
        self._bars_held: dict[str, int] = {s: 0 for s in symbols}

    @property
    def min_bars_required(self) -> int:
        return self.vol_lookback + 5

    # ------------------------------------------------------------------
    # Z-score and adaptive thresholds
    # ------------------------------------------------------------------

    @staticmethod
    def compute_zscore(close: pd.Series, window: int) -> pd.Series:
        """Compute rolling z-score of the close price.

        Parameters
        ----------
        close:
            Close price series.
        window:
            Rolling window.

        Returns
        -------
        pd.Series
            Z-score series.
        """
        rolling_mean = close.rolling(window=window, min_periods=window).mean()
        rolling_std = close.rolling(window=window, min_periods=window).std()
        rolling_std = rolling_std.replace(0, np.nan)
        return (close - rolling_mean) / rolling_std

    @staticmethod
    def compute_vol_ratio(close: pd.Series, short_window: int, long_window: int) -> pd.Series:
        """Compute the ratio of short-term to long-term volatility.

        Values > 1 indicate a high-vol regime, < 1 indicates low-vol.

        Parameters
        ----------
        close:
            Close price series.
        short_window:
            Short volatility window.
        long_window:
            Long volatility window.

        Returns
        -------
        pd.Series
            Vol ratio series.
        """
        log_ret = np.log(close / close.shift(1))
        short_vol = log_ret.rolling(window=short_window, min_periods=short_window).std()
        long_vol = log_ret.rolling(window=long_window, min_periods=long_window).std()
        long_vol = long_vol.replace(0, np.nan)
        return short_vol / long_vol

    def adaptive_threshold(self, vol_ratio: float) -> float:
        """Compute the adaptive entry z-score threshold.

        Parameters
        ----------
        vol_ratio:
            Current short/long volatility ratio.

        Returns
        -------
        float
            Adjusted entry threshold.
        """
        if np.isnan(vol_ratio):
            return self.base_entry_z
        return self.base_entry_z * (1.0 + self.vol_adjustment * (vol_ratio - 1.0))

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate_signal(self, symbol: str, bars: pd.DataFrame) -> Optional[SignalEvent]:
        """Generate z-score mean reversion signals with adaptive thresholds.

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
        z = self.compute_zscore(close, self.z_window)
        vol_r = self.compute_vol_ratio(close, self.z_window, self.vol_lookback)

        if z.isna().iloc[-1]:
            return None

        current_z = float(z.iloc[-1])
        current_price = float(close.iloc[-1])
        current_vol_r = float(vol_r.iloc[-1]) if not vol_r.isna().iloc[-1] else 1.0
        current_pos = self._position[symbol]

        entry_thresh = self.adaptive_threshold(current_vol_r)

        # --- Exit logic ---
        if current_pos is not None:
            self._bars_held[symbol] += 1

            # Time exit
            if self.max_holding_bars > 0 and self._bars_held[symbol] >= self.max_holding_bars:
                return self._exit(symbol, current_price, "time_exit", current_z, entry_thresh)

            # Stop-loss
            entry = self._entry_price[symbol]
            if current_pos == SignalDirection.LONG:
                if current_price <= entry * (1.0 - self.stop_loss_pct):
                    return self._exit(symbol, current_price, "stop_loss", current_z, entry_thresh)
            elif current_pos == SignalDirection.SHORT:
                if current_price >= entry * (1.0 + self.stop_loss_pct):
                    return self._exit(symbol, current_price, "stop_loss", current_z, entry_thresh)

            # Z-score reversion exit
            if current_pos == SignalDirection.LONG and current_z >= -self.base_exit_z:
                return self._exit(symbol, current_price, "z_revert", current_z, entry_thresh)
            if current_pos == SignalDirection.SHORT and current_z <= self.base_exit_z:
                return self._exit(symbol, current_price, "z_revert", current_z, entry_thresh)

        # --- Entry logic ---
        if current_pos is None:
            if current_z < -entry_thresh:
                strength = min(1.0, abs(current_z) / (entry_thresh * 2))
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
                        "z_score": round(current_z, 4),
                        "adaptive_threshold": round(entry_thresh, 4),
                        "vol_ratio": round(current_vol_r, 4),
                        "signal_type": "zscore_reversion",
                    },
                )

            if self.allow_short and current_z > entry_thresh:
                strength = min(1.0, abs(current_z) / (entry_thresh * 2))
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
                        "z_score": round(current_z, 4),
                        "adaptive_threshold": round(entry_thresh, 4),
                        "vol_ratio": round(current_vol_r, 4),
                        "signal_type": "zscore_reversion",
                    },
                )

        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _exit(
        self, symbol: str, price: float, reason: str,
        z: float, threshold: float,
    ) -> SignalEvent:
        """Build exit signal and reset position state."""
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
                "adaptive_threshold": round(threshold, 4),
                "signal_type": "zscore_reversion",
            },
        )
