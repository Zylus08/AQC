# AQC Strategy Development Guide

## Creating a Strategy

All strategies inherit from `BaseStrategy` and implement `generate_signal()`.

### Minimal Example

```python
from aqc.strategies.base_strategy import BaseStrategy
from aqc.backtester.event import SignalEvent, SignalDirection
from typing import Optional
import pandas as pd


class MeanReversionStrategy(BaseStrategy):
    """Simple Bollinger Band mean reversion."""

    def __init__(self, event_queue, symbols, period=20, std_dev=2.0):
        super().__init__(event_queue, symbols, strategy_id="bb_mean_reversion", lookback=period+5)
        self.period = period
        self.std_dev = std_dev

    @property
    def min_bars_required(self) -> int:
        return self.period + 1

    def generate_signal(self, symbol: str, bars: pd.DataFrame) -> Optional[SignalEvent]:
        from aqc.indicators.volatility import bollinger_bands
        close = bars["close"]
        upper, mid, lower = bollinger_bands(close, self.period, self.std_dev)

        if close.iloc[-1] < lower.iloc[-1]:
            return SignalEvent(symbol=symbol, strategy_id=self.strategy_id,
                               direction=SignalDirection.LONG, strength=1.0)
        elif close.iloc[-1] > upper.iloc[-1]:
            return SignalEvent(symbol=symbol, strategy_id=self.strategy_id,
                               direction=SignalDirection.EXIT, strength=1.0)
        return None
```

## Signal Strength

`SignalEvent.strength` is in `[-1.0, 1.0]`.

The portfolio uses it to scale position size:

```
order_quantity = default_quantity * abs(strength)
```

Use fractional strength for proportional sizing (e.g., `0.5` = half size).

## Available Indicators

```python
from aqc.indicators.moving_averages import sma, ema, wma, dema, hull_ma
from aqc.indicators.momentum import rsi, macd, stochastic, rate_of_change
from aqc.indicators.volatility import bollinger_bands, atr, historical_volatility
```

All functions take `pd.Series` and return `pd.Series`.

## Multi-Symbol Strategies

The `symbols` list can contain multiple instruments.
`on_market_event()` is called once per bar per symbol.
The `_bar_buffer` is keyed by symbol automatically.

## Signal Directions

| Direction | Meaning |
|-----------|---------|
| `LONG` | Buy / go long |
| `SHORT` | Sell short |
| `EXIT` | Close existing long position |
| `HOLD` | Do nothing (no order emitted) |
