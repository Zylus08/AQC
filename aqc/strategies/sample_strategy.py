"""
aqc/strategies/sample_strategy.py
===================================
Concrete strategy implementations for testing the AQC framework.

This module ships two ready-to-run strategies:

1. :class:`SMACrossoverStrategy` — Classic dual-SMA crossover.
2. :class:`RSIMeanReversionStrategy` — RSI-based mean reversion.

Both serve as working examples and integration tests for the full
data → strategy → signal → order → fill pipeline.

Author: AQC Team
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from aqc.backtester.event import SignalDirection, SignalEvent
from aqc.indicators.moving_averages import sma, ema
from aqc.indicators.momentum import rsi
from aqc.strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class SMACrossoverStrategy(BaseStrategy):
    """Dual Simple Moving Average Crossover Strategy.

    **Logic**

    * **Long entry**: fast SMA crosses *above* slow SMA.
    * **Exit long**: fast SMA crosses *below* slow SMA.

    The strategy is always either long or flat (no shorting).

    Parameters
    ----------
    event_queue:
        Shared event queue.
    symbols:
        Instruments to trade.
    fast_period:
        Fast SMA period (default 20).
    slow_period:
        Slow SMA period (default 50).
    strategy_id:
        Strategy label (default ``"sma_crossover"``).

    Examples
    --------
    >>> strategy = SMACrossoverStrategy(
    ...     event_queue=eq,
    ...     symbols=["AAPL"],
    ...     fast_period=10,
    ...     slow_period=30,
    ... )
    """

    def __init__(
        self,
        event_queue,
        symbols: list[str],
        fast_period: int = 20,
        slow_period: int = 50,
        strategy_id: str = "sma_crossover",
    ) -> None:
        super().__init__(
            event_queue=event_queue,
            symbols=symbols,
            strategy_id=strategy_id,
            lookback=slow_period + 10,  # enough buffer to compute both SMAs
        )
        self.fast_period = fast_period
        self.slow_period = slow_period
        self._position: dict[str, Optional[SignalDirection]] = {s: None for s in symbols}

    @property
    def min_bars_required(self) -> int:
        """Need at least ``slow_period + 1`` bars to detect a crossover."""
        return self.slow_period + 1

    def generate_signal(self, symbol: str, bars: pd.DataFrame) -> Optional[SignalEvent]:
        """Check for SMA crossover and emit a signal.

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
        fast = sma(close, self.fast_period)
        slow = sma(close, self.slow_period)

        if fast.isna().iloc[-1] or slow.isna().iloc[-1]:
            return None

        current_fast = fast.iloc[-1]
        current_slow = slow.iloc[-1]
        prev_fast = fast.iloc[-2]
        prev_slow = slow.iloc[-2]

        current_position = self._position[symbol]

        # Golden cross: fast crosses above slow
        if prev_fast <= prev_slow and current_fast > current_slow:
            if current_position != SignalDirection.LONG:
                logger.debug(
                    "%s LONG signal: fast=%.4f > slow=%.4f",
                    symbol,
                    current_fast,
                    current_slow,
                )
                self._position[symbol] = SignalDirection.LONG
                return SignalEvent(
                    symbol=symbol,
                    strategy_id=self.strategy_id,
                    direction=SignalDirection.LONG,
                    strength=1.0,
                    suggested_price=bars["close"].iloc[-1],
                    metadata={
                        "fast_sma": round(current_fast, 4),
                        "slow_sma": round(current_slow, 4),
                    },
                )

        # Death cross: fast crosses below slow
        elif prev_fast >= prev_slow and current_fast < current_slow:
            if current_position == SignalDirection.LONG:
                logger.debug(
                    "%s EXIT signal: fast=%.4f < slow=%.4f",
                    symbol,
                    current_fast,
                    current_slow,
                )
                self._position[symbol] = None
                return SignalEvent(
                    symbol=symbol,
                    strategy_id=self.strategy_id,
                    direction=SignalDirection.EXIT,
                    strength=1.0,
                    suggested_price=bars["close"].iloc[-1],
                    metadata={
                        "fast_sma": round(current_fast, 4),
                        "slow_sma": round(current_slow, 4),
                    },
                )

        return None


class RSIMeanReversionStrategy(BaseStrategy):
    """RSI-Based Mean Reversion Strategy.

    **Logic**

    * **Long entry**: RSI falls below *oversold* threshold.
    * **Long exit**: RSI rises above *overbought* threshold.
    * **Short entry**: RSI rises above *overbought* threshold (optional).
    * **Short exit**: RSI falls below *oversold* threshold.

    Parameters
    ----------
    event_queue:
        Shared event queue.
    symbols:
        Instruments to trade.
    rsi_period:
        RSI calculation period (default 14).
    oversold:
        RSI level considered oversold — triggers long entry (default 30).
    overbought:
        RSI level considered overbought — triggers exit / short (default 70).
    allow_short:
        If ``True``, the strategy also shorts when overbought (default False).
    strategy_id:
        Strategy label.

    Examples
    --------
    >>> strategy = RSIMeanReversionStrategy(
    ...     event_queue=eq,
    ...     symbols=["AAPL"],
    ...     rsi_period=14,
    ...     oversold=30,
    ...     overbought=70,
    ... )
    """

    def __init__(
        self,
        event_queue,
        symbols: list[str],
        rsi_period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        allow_short: bool = False,
        strategy_id: str = "rsi_mean_reversion",
    ) -> None:
        super().__init__(
            event_queue=event_queue,
            symbols=symbols,
            strategy_id=strategy_id,
            lookback=rsi_period * 3,
        )
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.allow_short = allow_short
        self._position: dict[str, Optional[SignalDirection]] = {s: None for s in symbols}

    @property
    def min_bars_required(self) -> int:
        return self.rsi_period * 2

    def generate_signal(self, symbol: str, bars: pd.DataFrame) -> Optional[SignalEvent]:
        """Compute RSI and check for mean-reversion signals.

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
        rsi_series = rsi(close, self.rsi_period)

        if rsi_series.isna().iloc[-1]:
            return None

        current_rsi = rsi_series.iloc[-1]
        prev_rsi = rsi_series.iloc[-2]
        current_pos = self._position[symbol]
        current_price = close.iloc[-1]

        # --- Long entry: RSI crosses below oversold ---
        if prev_rsi >= self.oversold and current_rsi < self.oversold:
            if current_pos != SignalDirection.LONG:
                self._position[symbol] = SignalDirection.LONG
                return SignalEvent(
                    symbol=symbol,
                    strategy_id=self.strategy_id,
                    direction=SignalDirection.LONG,
                    strength=min(1.0, (self.oversold - current_rsi) / self.oversold),
                    suggested_price=current_price,
                    metadata={"rsi": round(current_rsi, 2)},
                )

        # --- Exit long: RSI crosses above overbought ---
        elif current_rsi > self.overbought and current_pos == SignalDirection.LONG:
            self._position[symbol] = None
            return SignalEvent(
                symbol=symbol,
                strategy_id=self.strategy_id,
                direction=SignalDirection.EXIT,
                strength=1.0,
                suggested_price=current_price,
                metadata={"rsi": round(current_rsi, 2)},
            )

        # --- Short entry (optional): RSI crosses above overbought ---
        elif (
            self.allow_short
            and prev_rsi <= self.overbought
            and current_rsi > self.overbought
            and current_pos != SignalDirection.SHORT
        ):
            self._position[symbol] = SignalDirection.SHORT
            return SignalEvent(
                symbol=symbol,
                strategy_id=self.strategy_id,
                direction=SignalDirection.SHORT,
                strength=min(1.0, (current_rsi - self.overbought) / (100 - self.overbought)),
                suggested_price=current_price,
                metadata={"rsi": round(current_rsi, 2)},
            )

        # --- Exit short: RSI crosses below oversold ---
        elif (
            self.allow_short
            and current_rsi < self.oversold
            and current_pos == SignalDirection.SHORT
        ):
            self._position[symbol] = None
            return SignalEvent(
                symbol=symbol,
                strategy_id=self.strategy_id,
                direction=SignalDirection.EXIT,
                strength=1.0,
                suggested_price=current_price,
                metadata={"rsi": round(current_rsi, 2)},
            )

        return None


class EMAMomentumStrategy(BaseStrategy):
    """EMA-based momentum strategy with trend filter.

    **Logic**

    * Trend filter: price must be above the long EMA to allow longs.
    * Entry: short EMA > medium EMA > long EMA (aligned trend).
    * Exit: short EMA < medium EMA.

    Parameters
    ----------
    event_queue:
        Shared event queue.
    symbols:
        Instruments to trade.
    short_period:
        Short EMA period (default 9).
    medium_period:
        Medium EMA period (default 21).
    long_period:
        Long EMA (trend filter) period (default 50).
    strategy_id:
        Strategy label.
    """

    def __init__(
        self,
        event_queue,
        symbols: list[str],
        short_period: int = 9,
        medium_period: int = 21,
        long_period: int = 50,
        strategy_id: str = "ema_momentum",
    ) -> None:
        super().__init__(
            event_queue=event_queue,
            symbols=symbols,
            strategy_id=strategy_id,
            lookback=long_period + 10,
        )
        self.short_period = short_period
        self.medium_period = medium_period
        self.long_period = long_period
        self._position: dict[str, Optional[SignalDirection]] = {s: None for s in symbols}

    @property
    def min_bars_required(self) -> int:
        return self.long_period + 2

    def generate_signal(self, symbol: str, bars: pd.DataFrame) -> Optional[SignalEvent]:
        """Check for EMA alignment momentum signals."""
        close = bars["close"]
        e_short = ema(close, self.short_period)
        e_medium = ema(close, self.medium_period)
        e_long = ema(close, self.long_period)

        if any(s.isna().iloc[-1] for s in [e_short, e_medium, e_long]):
            return None

        cs, cm, cl = e_short.iloc[-1], e_medium.iloc[-1], e_long.iloc[-1]
        ps, pm = e_short.iloc[-2], e_medium.iloc[-2]
        price = close.iloc[-1]
        current_pos = self._position[symbol]

        # Entry: EMAs aligned and price above trend filter
        if cs > cm > cl and price > cl and not (ps > pm):
            if current_pos != SignalDirection.LONG:
                self._position[symbol] = SignalDirection.LONG
                return SignalEvent(
                    symbol=symbol,
                    strategy_id=self.strategy_id,
                    direction=SignalDirection.LONG,
                    strength=1.0,
                    suggested_price=price,
                )

        # Exit: short EMA drops below medium EMA
        elif cs < cm and current_pos == SignalDirection.LONG:
            self._position[symbol] = None
            return SignalEvent(
                symbol=symbol,
                strategy_id=self.strategy_id,
                direction=SignalDirection.EXIT,
                strength=1.0,
                suggested_price=price,
            )

        return None
