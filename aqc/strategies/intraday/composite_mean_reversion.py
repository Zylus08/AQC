"""
aqc/strategies/intraday/composite_mean_reversion.py
====================================================
Composite Multi-Signal Mean Reversion Strategy.

Alpha thesis
------------
Individual mean-reversion signals are noisy.  Combining multiple
orthogonal signals into a composite alpha score produces more robust
trades with higher Sharpe ratios.

This strategy blends three sub-signals:

1. **VWAP deviation z-score** — price deviation from VWAP.
2. **Volume exhaustion score** — abnormal volume + failed breakout.
3. **Rolling z-score** — standard z-score with vol-adaptive thresholds.

Each sub-signal produces a normalised score in ``[-1, 1]``:
* Negative = bearish (price extended upward, expect reversion down).
* Positive = bullish (price extended downward, expect reversion up).

The composite alpha is a weighted average:

    alpha = w_vwap * vwap_score + w_volume * volume_score + w_zscore * z_score

Entries occur when ``|alpha| > composite_threshold`` and the direction
is consistent across at least ``min_signals`` sub-signals.

Features
--------
* Per-signal configurable weights.
* Minimum signal agreement filter.
* Configurable holding periods and stop-loss.
* Rich metadata for research analysis (per-signal breakdown).

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from aqc.backtester.event import SignalDirection, SignalEvent
from aqc.strategies.base_strategy import BaseStrategy
from aqc.strategies.intraday.vwap_reversion import VWAPReversionStrategy
from aqc.strategies.intraday.zscore_reversion import ZScoreReversionStrategy
from aqc.strategies.intraday.volume_exhaustion import VolumeExhaustionStrategy

logger = logging.getLogger(__name__)


class CompositeMeanReversionStrategy(BaseStrategy):
    """Multi-signal composite mean reversion strategy.

    Blends VWAP deviation, volume exhaustion, and z-score signals into
    a single alpha score with configurable weights and agreement filters.

    Parameters
    ----------
    event_queue:
        Shared event queue.
    symbols:
        Instruments to trade.
    w_vwap:
        Weight for VWAP deviation signal (default 0.4).
    w_volume:
        Weight for volume exhaustion signal (default 0.3).
    w_zscore:
        Weight for z-score signal (default 0.3).
    composite_threshold:
        Min |alpha| to enter a position (default 0.3).
    exit_threshold:
        Alpha threshold to exit (default 0.1).
    min_signals:
        Min number of sub-signals that must agree (default 2).
    vwap_window:
        Rolling window for VWAP std (default 20).
    vwap_entry_z:
        VWAP entry z-score threshold (default 2.0).
    z_window:
        Rolling window for z-score (default 20).
    z_entry:
        Z-score entry threshold (default 2.0).
    vol_window:
        Volume average window (default 20).
    vol_spike_mult:
        Volume spike multiplier (default 2.0).
    breakout_window:
        Lookback for failed breakouts (default 10).
    max_holding_bars:
        Force exit after this many bars (default 20, 0 = disabled).
    stop_loss_pct:
        Stop-loss fraction from entry (default 0.02).
    strategy_id:
        Strategy label.

    Examples
    --------
    >>> strategy = CompositeMeanReversionStrategy(
    ...     event_queue=eq, symbols=["AAPL"],
    ...     w_vwap=0.4, w_volume=0.3, w_zscore=0.3,
    ... )
    """

    def __init__(
        self,
        event_queue,
        symbols: list[str],
        w_vwap: float = 0.4,
        w_volume: float = 0.3,
        w_zscore: float = 0.3,
        composite_threshold: float = 0.3,
        exit_threshold: float = 0.1,
        min_signals: int = 2,
        vwap_window: int = 20,
        vwap_entry_z: float = 2.0,
        z_window: int = 20,
        z_entry: float = 2.0,
        vol_window: int = 20,
        vol_spike_mult: float = 2.0,
        breakout_window: int = 10,
        max_holding_bars: int = 20,
        stop_loss_pct: float = 0.02,
        strategy_id: str = "composite_mean_reversion",
    ) -> None:
        lookback_need = max(vwap_window, z_window, vol_window, breakout_window) * 4
        super().__init__(
            event_queue=event_queue,
            symbols=symbols,
            strategy_id=strategy_id,
            lookback=lookback_need,
        )
        # Signal weights (normalised)
        total_w = w_vwap + w_volume + w_zscore
        self.w_vwap = w_vwap / total_w
        self.w_volume = w_volume / total_w
        self.w_zscore = w_zscore / total_w

        self.composite_threshold = composite_threshold
        self.exit_threshold = exit_threshold
        self.min_signals = min_signals

        # Sub-signal parameters
        self.vwap_window = vwap_window
        self.vwap_entry_z = vwap_entry_z
        self.z_window = z_window
        self.z_entry = z_entry
        self.vol_window = vol_window
        self.vol_spike_mult = vol_spike_mult
        self.breakout_window = breakout_window

        # Risk parameters
        self.max_holding_bars = max_holding_bars
        self.stop_loss_pct = stop_loss_pct

        # State
        self._position: dict[str, Optional[SignalDirection]] = {s: None for s in symbols}
        self._entry_price: dict[str, float] = {s: 0.0 for s in symbols}
        self._bars_held: dict[str, int] = {s: 0 for s in symbols}

    @property
    def min_bars_required(self) -> int:
        return max(self.vwap_window, self.z_window, self.vol_window, self.breakout_window) + 10

    # ------------------------------------------------------------------
    # Sub-signal computations
    # ------------------------------------------------------------------

    def compute_vwap_signal(self, bars: pd.DataFrame) -> float:
        """Compute the VWAP deviation signal as a normalised score in [-1, 1].

        Negative = bullish (price below VWAP, expect reversion up).
        Positive = bearish (price above VWAP, expect reversion down).

        We invert the z-score to produce a *reversion* direction:
        ``signal = -clamp(z / entry_z, -1, 1)``

        Parameters
        ----------
        bars:
            OHLCV bar window.

        Returns
        -------
        float
            VWAP deviation signal score.
        """
        vwap = VWAPReversionStrategy.compute_vwap(bars)
        z = VWAPReversionStrategy.compute_vwap_zscore(
            bars["close"], vwap, self.vwap_window
        )

        if z.isna().iloc[-1]:
            return 0.0

        current_z = float(z.iloc[-1])
        # Invert: high positive z -> we expect reversion down -> bullish reversion signal is negative
        normalised = -current_z / self.vwap_entry_z
        return float(np.clip(normalised, -1.0, 1.0))

    def compute_volume_signal(self, bars: pd.DataFrame) -> float:
        """Compute the volume exhaustion signal score in [-1, 1].

        Detects volume spikes + failed breakouts. Signal is directional
        based on which side the breakout failed:
        * Failed up breakout -> negative (bearish, expect reversion down).
        * Failed down breakout -> positive (bullish, expect reversion up).

        Parameters
        ----------
        bars:
            OHLCV bar window.

        Returns
        -------
        float
            Volume exhaustion signal score.
        """
        vol_spike = VolumeExhaustionStrategy.detect_volume_spike(
            bars["volume"], self.vol_window, self.vol_spike_mult
        )
        failed_up, failed_down = VolumeExhaustionStrategy.detect_failed_breakout(
            bars["high"], bars["low"], bars["close"], self.breakout_window
        )

        is_spike = bool(vol_spike.iloc[-1]) if not vol_spike.isna().iloc[-1] else False
        if not is_spike:
            return 0.0

        is_failed_up = bool(failed_up.iloc[-1]) if not failed_up.isna().iloc[-1] else False
        is_failed_down = bool(failed_down.iloc[-1]) if not failed_down.isna().iloc[-1] else False

        # Compute intensity from vol ratio
        avg_vol = bars["volume"].rolling(self.vol_window).mean().iloc[-1]
        current_vol = float(bars["volume"].iloc[-1])
        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
        intensity = min(1.0, (vol_ratio - 1.0) / self.vol_spike_mult)

        if is_failed_up:
            return -intensity  # bearish reversal
        elif is_failed_down:
            return intensity  # bullish reversal
        return 0.0

    def compute_zscore_signal(self, bars: pd.DataFrame) -> float:
        """Compute the rolling z-score signal in [-1, 1].

        Same inversion as VWAP: ``signal = -clamp(z / entry_z, -1, 1)``

        Parameters
        ----------
        bars:
            OHLCV bar window.

        Returns
        -------
        float
            Z-score signal score.
        """
        z = ZScoreReversionStrategy.compute_zscore(bars["close"], self.z_window)

        if z.isna().iloc[-1]:
            return 0.0

        current_z = float(z.iloc[-1])
        normalised = -current_z / self.z_entry
        return float(np.clip(normalised, -1.0, 1.0))

    # ------------------------------------------------------------------
    # Composite alpha
    # ------------------------------------------------------------------

    def compute_composite_alpha(
        self, bars: pd.DataFrame
    ) -> tuple[float, float, float, float]:
        """Compute the composite alpha from all sub-signals.

        Parameters
        ----------
        bars:
            OHLCV bar window.

        Returns
        -------
        tuple[float, float, float, float]
            ``(composite_alpha, vwap_signal, volume_signal, zscore_signal)``
        """
        vwap_sig = self.compute_vwap_signal(bars)
        vol_sig = self.compute_volume_signal(bars)
        z_sig = self.compute_zscore_signal(bars)

        alpha = (
            self.w_vwap * vwap_sig
            + self.w_volume * vol_sig
            + self.w_zscore * z_sig
        )

        return alpha, vwap_sig, vol_sig, z_sig

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate_signal(self, symbol: str, bars: pd.DataFrame) -> Optional[SignalEvent]:
        """Generate composite mean reversion signals.

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
        current_pos = self._position[symbol]

        alpha, vwap_sig, vol_sig, z_sig = self.compute_composite_alpha(bars)

        # --- Exit logic ---
        if current_pos is not None:
            self._bars_held[symbol] += 1

            # Time exit
            if self.max_holding_bars > 0 and self._bars_held[symbol] >= self.max_holding_bars:
                return self._exit(
                    symbol, current_price, "time_exit",
                    alpha, vwap_sig, vol_sig, z_sig,
                )

            # Stop-loss
            entry = self._entry_price[symbol]
            if current_pos == SignalDirection.LONG:
                if current_price <= entry * (1.0 - self.stop_loss_pct):
                    return self._exit(
                        symbol, current_price, "stop_loss",
                        alpha, vwap_sig, vol_sig, z_sig,
                    )
            elif current_pos == SignalDirection.SHORT:
                if current_price >= entry * (1.0 + self.stop_loss_pct):
                    return self._exit(
                        symbol, current_price, "stop_loss",
                        alpha, vwap_sig, vol_sig, z_sig,
                    )

            # Alpha reversion exit
            if current_pos == SignalDirection.LONG and alpha < self.exit_threshold:
                return self._exit(
                    symbol, current_price, "alpha_revert",
                    alpha, vwap_sig, vol_sig, z_sig,
                )
            if current_pos == SignalDirection.SHORT and alpha > -self.exit_threshold:
                return self._exit(
                    symbol, current_price, "alpha_revert",
                    alpha, vwap_sig, vol_sig, z_sig,
                )

        # --- Entry logic ---
        if current_pos is not None:
            return None

        # Count agreeing signals
        signals = [vwap_sig, vol_sig, z_sig]
        n_bullish = sum(1 for s in signals if s > 0.1)
        n_bearish = sum(1 for s in signals if s < -0.1)

        # Bullish composite entry
        if alpha > self.composite_threshold and n_bullish >= self.min_signals:
            strength = min(1.0, alpha)
            self._position[symbol] = SignalDirection.LONG
            self._entry_price[symbol] = current_price
            self._bars_held[symbol] = 0
            return SignalEvent(
                symbol=symbol,
                strategy_id=self.strategy_id,
                direction=SignalDirection.LONG,
                strength=strength,
                suggested_price=current_price,
                metadata=self._build_metadata(
                    "entry", alpha, vwap_sig, vol_sig, z_sig, n_bullish,
                ),
            )

        # Bearish composite entry
        if alpha < -self.composite_threshold and n_bearish >= self.min_signals:
            strength = min(1.0, abs(alpha))
            self._position[symbol] = SignalDirection.SHORT
            self._entry_price[symbol] = current_price
            self._bars_held[symbol] = 0
            return SignalEvent(
                symbol=symbol,
                strategy_id=self.strategy_id,
                direction=SignalDirection.SHORT,
                strength=strength,
                suggested_price=current_price,
                metadata=self._build_metadata(
                    "entry", alpha, vwap_sig, vol_sig, z_sig, n_bearish,
                ),
            )

        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _exit(
        self, symbol: str, price: float, reason: str,
        alpha: float, vwap_sig: float, vol_sig: float, z_sig: float,
    ) -> SignalEvent:
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
            metadata=self._build_metadata(
                reason, alpha, vwap_sig, vol_sig, z_sig, 0,
            ),
        )

    @staticmethod
    def _build_metadata(
        action: str,
        alpha: float,
        vwap_sig: float,
        vol_sig: float,
        z_sig: float,
        n_agree: int,
    ) -> dict:
        """Build rich metadata dict for research analysis.

        Parameters
        ----------
        action / alpha / vwap_sig / vol_sig / z_sig / n_agree:
            Signal state.

        Returns
        -------
        dict
        """
        return {
            "signal_type": "composite_mean_reversion",
            "action": action,
            "composite_alpha": round(alpha, 4),
            "vwap_signal": round(vwap_sig, 4),
            "volume_signal": round(vol_sig, 4),
            "zscore_signal": round(z_sig, 4),
            "n_agreeing_signals": n_agree,
        }
