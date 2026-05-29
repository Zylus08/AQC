"""
aqc/strategies/intraday/volume_exhaustion.py
=============================================
Volume Exhaustion / Failed Breakout Mean Reversion Strategy.

Alpha thesis
------------
Strong directional moves accompanied by abnormally high volume often
signal the *exhaustion* of the current trend.  When a breakout attempt
occurs on extreme volume but fails to sustain, price typically reverts
quickly.

The strategy detects three converging conditions:

1. **Volume spike**: volume exceeds ``spike_mult`` x the rolling average.
2. **Failed breakout**: price makes a new short-term high/low but then
   closes back inside the recent range (wick rejection).
3. **Exhaustion reversal**: the bar's close is in the opposite half of
   its own range (upper-tail rejection for longs, lower-tail for shorts).

All three must fire simultaneously for a high-confidence signal.  The
strategy also supports a *relaxed mode* that requires only conditions
1+2 (higher frequency, lower win rate).

Risk management
---------------
* Configurable ``max_holding_bars`` for time-based exits.
* ``stop_loss_pct`` from entry price.
* Exit when volume normalises (``< exit_volume_ratio * avg_volume``).

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


class VolumeExhaustionStrategy(BaseStrategy):
    """Volume exhaustion / failed breakout reversal strategy.

    Parameters
    ----------
    event_queue:
        Shared event queue.
    symbols:
        Instruments to trade.
    volume_window:
        Rolling window for average volume (default 20).
    spike_mult:
        Volume must exceed ``spike_mult * avg_volume`` (default 2.5).
    breakout_window:
        Lookback for recent high/low range (default 10).
    wick_ratio:
        Min wick-to-body ratio for exhaustion candle (default 0.6).
    max_holding_bars:
        Force exit after this many bars (default 15, 0 = disabled).
    stop_loss_pct:
        Stop-loss fraction from entry (default 0.015 = 1.5%).
    exit_volume_ratio:
        Exit if volume drops below this fraction of avg (default 0.8).
    require_wick_rejection:
        If True (default), all 3 conditions must be met.
        If False, only volume spike + failed breakout are needed.
    strategy_id:
        Strategy label.

    Examples
    --------
    >>> strategy = VolumeExhaustionStrategy(
    ...     event_queue=eq, symbols=["AAPL"],
    ...     spike_mult=2.5, breakout_window=10,
    ... )
    """

    def __init__(
        self,
        event_queue,
        symbols: list[str],
        volume_window: int = 20,
        spike_mult: float = 2.5,
        breakout_window: int = 10,
        wick_ratio: float = 0.6,
        max_holding_bars: int = 15,
        stop_loss_pct: float = 0.015,
        exit_volume_ratio: float = 0.8,
        require_wick_rejection: bool = True,
        strategy_id: str = "volume_exhaustion",
    ) -> None:
        super().__init__(
            event_queue=event_queue,
            symbols=symbols,
            strategy_id=strategy_id,
            lookback=max(volume_window, breakout_window) * 3,
        )
        self.volume_window = volume_window
        self.spike_mult = spike_mult
        self.breakout_window = breakout_window
        self.wick_ratio = wick_ratio
        self.max_holding_bars = max_holding_bars
        self.stop_loss_pct = stop_loss_pct
        self.exit_volume_ratio = exit_volume_ratio
        self.require_wick_rejection = require_wick_rejection

        self._position: dict[str, Optional[SignalDirection]] = {s: None for s in symbols}
        self._entry_price: dict[str, float] = {s: 0.0 for s in symbols}
        self._bars_held: dict[str, int] = {s: 0 for s in symbols}

    @property
    def min_bars_required(self) -> int:
        return max(self.volume_window, self.breakout_window) + 5

    # ------------------------------------------------------------------
    # Signal detectors
    # ------------------------------------------------------------------

    @staticmethod
    def detect_volume_spike(
        volume: pd.Series, window: int, mult: float
    ) -> pd.Series:
        """Detect bars where volume exceeds ``mult`` x rolling mean.

        Parameters
        ----------
        volume:
            Volume series.
        window:
            Rolling window for average.
        mult:
            Spike multiplier.

        Returns
        -------
        pd.Series
            Boolean series — ``True`` on spike bars.
        """
        avg_vol = volume.rolling(window=window, min_periods=window).mean()
        return volume > (avg_vol * mult)

    @staticmethod
    def detect_failed_breakout(
        high: pd.Series, low: pd.Series, close: pd.Series, window: int
    ) -> tuple[pd.Series, pd.Series]:
        """Detect failed breakout attempts.

        A *failed breakout up* occurs when:
        - Current high > rolling max high (breakout attempt)
        - Close < rolling max high (failed to hold)

        A *failed breakout down* occurs when:
        - Current low < rolling min low (breakdown attempt)
        - Close > rolling min low (failed to hold)

        Parameters
        ----------
        high / low / close:
            Price series.
        window:
            Lookback for rolling max/min.

        Returns
        -------
        tuple[pd.Series, pd.Series]
            ``(failed_up, failed_down)`` boolean series.
        """
        rolling_high = high.rolling(window=window, min_periods=window).max()
        rolling_low = low.rolling(window=window, min_periods=window).min()

        # Shift by 1 to compare current bar against the *prior* range
        prev_high = rolling_high.shift(1)
        prev_low = rolling_low.shift(1)

        failed_up = (high > prev_high) & (close < prev_high)
        failed_down = (low < prev_low) & (close > prev_low)

        return failed_up, failed_down

    @staticmethod
    def detect_wick_rejection(
        open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series,
        wick_ratio: float,
    ) -> tuple[pd.Series, pd.Series]:
        """Detect candles with long upper/lower wicks (rejection).

        An upper wick rejection (bearish): upper wick > ``wick_ratio`` of range.
        A lower wick rejection (bullish): lower wick > ``wick_ratio`` of range.

        Parameters
        ----------
        open_ / high / low / close:
            Price series.
        wick_ratio:
            Minimum wick-to-range ratio (0 to 1).

        Returns
        -------
        tuple[pd.Series, pd.Series]
            ``(upper_rejection, lower_rejection)`` boolean series.
        """
        total_range = (high - low).replace(0, np.nan)
        body_top = pd.concat([open_, close], axis=1).max(axis=1)
        body_bottom = pd.concat([open_, close], axis=1).min(axis=1)

        upper_wick = (high - body_top) / total_range
        lower_wick = (body_bottom - low) / total_range

        upper_rejection = upper_wick > wick_ratio
        lower_rejection = lower_wick > wick_ratio

        return upper_rejection, lower_rejection

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate_signal(self, symbol: str, bars: pd.DataFrame) -> Optional[SignalEvent]:
        """Generate volume exhaustion reversal signals.

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
        current_price = float(bars["close"].iloc[-1])
        current_vol = float(bars["volume"].iloc[-1])
        current_pos = self._position[symbol]

        # --- Exit logic ---
        if current_pos is not None:
            self._bars_held[symbol] += 1
            exit_signal = self._check_exit(symbol, bars, current_price, current_vol)
            if exit_signal is not None:
                return exit_signal

        # --- Entry logic ---
        if current_pos is not None:
            return None

        vol_spike = self.detect_volume_spike(bars["volume"], self.volume_window, self.spike_mult)
        failed_up, failed_down = self.detect_failed_breakout(
            bars["high"], bars["low"], bars["close"], self.breakout_window
        )
        upper_rej, lower_rej = self.detect_wick_rejection(
            bars["open"], bars["high"], bars["low"], bars["close"],
            self.wick_ratio,
        )

        # Current bar values
        is_spike = bool(vol_spike.iloc[-1]) if not vol_spike.isna().iloc[-1] else False
        is_failed_up = bool(failed_up.iloc[-1]) if not failed_up.isna().iloc[-1] else False
        is_failed_down = bool(failed_down.iloc[-1]) if not failed_down.isna().iloc[-1] else False
        is_upper_rej = bool(upper_rej.iloc[-1]) if not upper_rej.isna().iloc[-1] else False
        is_lower_rej = bool(lower_rej.iloc[-1]) if not lower_rej.isna().iloc[-1] else False

        if not is_spike:
            return None

        # Volume ratio for signal strength
        avg_vol = bars["volume"].rolling(self.volume_window).mean().iloc[-1]
        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0

        # Bearish exhaustion -> go long (reversal from failed up breakout)
        # Counter-intuitive: a failed breakout UP means price will revert DOWN,
        # but the *exhaustion* of the upward move implies sellers stepping in.
        # We enter SHORT after a failed breakout up.
        if is_failed_up and (is_upper_rej or not self.require_wick_rejection):
            strength = min(1.0, vol_ratio / (self.spike_mult * 2))
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
                    "vol_ratio": round(vol_ratio, 2),
                    "failed_breakout": "up",
                    "wick_rejection": is_upper_rej,
                    "signal_type": "volume_exhaustion",
                },
            )

        # Bullish exhaustion -> go long (failed breakdown)
        if is_failed_down and (is_lower_rej or not self.require_wick_rejection):
            strength = min(1.0, vol_ratio / (self.spike_mult * 2))
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
                    "vol_ratio": round(vol_ratio, 2),
                    "failed_breakout": "down",
                    "wick_rejection": is_lower_rej,
                    "signal_type": "volume_exhaustion",
                },
            )

        return None

    # ------------------------------------------------------------------
    # Exit checks
    # ------------------------------------------------------------------

    def _check_exit(
        self, symbol: str, bars: pd.DataFrame, price: float, volume: float
    ) -> Optional[SignalEvent]:
        """Check time, stop-loss, and volume-normalisation exits.

        Parameters
        ----------
        symbol / bars / price / volume:
            Current state.

        Returns
        -------
        SignalEvent | None
        """
        current_pos = self._position[symbol]
        entry = self._entry_price[symbol]
        held = self._bars_held[symbol]

        # Time exit
        if self.max_holding_bars > 0 and held >= self.max_holding_bars:
            return self._exit(symbol, price, "time_exit")

        # Stop-loss
        if current_pos == SignalDirection.LONG:
            if price <= entry * (1.0 - self.stop_loss_pct):
                return self._exit(symbol, price, "stop_loss")
        elif current_pos == SignalDirection.SHORT:
            if price >= entry * (1.0 + self.stop_loss_pct):
                return self._exit(symbol, price, "stop_loss")

        # Volume normalisation exit
        avg_vol = bars["volume"].rolling(self.volume_window).mean().iloc[-1]
        if not np.isnan(avg_vol) and avg_vol > 0:
            if volume < avg_vol * self.exit_volume_ratio:
                return self._exit(symbol, price, "volume_normalised")

        return None

    def _exit(self, symbol: str, price: float, reason: str) -> SignalEvent:
        """Build exit signal and reset state."""
        self._position[symbol] = None
        self._entry_price[symbol] = 0.0
        self._bars_held[symbol] = 0
        return SignalEvent(
            symbol=symbol,
            strategy_id=self.strategy_id,
            direction=SignalDirection.EXIT,
            strength=1.0,
            suggested_price=price,
            metadata={"exit_reason": reason, "signal_type": "volume_exhaustion"},
        )
