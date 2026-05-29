# Intraday Mean Reversion Research Guide

## Overview

The AQC Intraday Mean Reversion Suite provides four research-grade strategies
that trade the statistical tendency of prices to revert toward equilibrium
levels after temporary dislocations.

```
aqc/strategies/intraday/
├── vwap_reversion.py             # VWAP deviation z-score
├── volume_exhaustion.py          # Volume spike + failed breakout
├── zscore_reversion.py           # Rolling z-score with adaptive thresholds
└── composite_mean_reversion.py   # Multi-signal alpha composite
```

All four strategies:
- Inherit from `BaseStrategy` and integrate with the existing `BacktestEngine`
- Support configurable entry/exit thresholds, holding periods, and stop-losses
- Emit rich metadata for post-trade research analysis
- Are compatible with the Walk-Forward Optimisation framework

---

## Strategy 1: VWAP Deviation Reversion

### Alpha Thesis

Intraday prices revert toward the Volume-Weighted Average Price.
When the z-score of `(price - VWAP) / rolling_std` exceeds a threshold,
the dislocation is statistically significant and likely to revert.

### Signal Computation

```
VWAP = cumsum(typical_price * volume) / cumsum(volume)
z = (close - VWAP) / rolling_std(close - VWAP, window)

Entry LONG:  z < -entry_threshold
Entry SHORT: z > +entry_threshold
Exit:        |z| < exit_threshold
```

### Usage

```python
from aqc.strategies.intraday import VWAPReversionStrategy

strategy = VWAPReversionStrategy(
    event_queue=eq,
    symbols=["AAPL"],
    entry_threshold=2.0,     # z-score to enter
    exit_threshold=0.5,      # z-score to exit
    rolling_window=20,       # std calculation window
    max_holding_bars=20,     # force exit after 20 bars
    stop_loss_pct=0.02,      # 2% stop loss
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `entry_threshold` | float | 2.0 | Z-score magnitude for entry |
| `exit_threshold` | float | 0.5 | Z-score magnitude for exit |
| `rolling_window` | int | 20 | Window for rolling std |
| `max_holding_bars` | int | 20 | Max bars before force exit |
| `stop_loss_pct` | float | 0.02 | Stop-loss from entry price |

---

## Strategy 2: Volume Exhaustion

### Alpha Thesis

Extreme volume events often mark trend exhaustion.  A breakout attempt
on abnormal volume that fails to sustain signals a high-probability
reversal.  The strategy requires triple confluence:

1. **Volume spike**: volume > `spike_mult` x rolling average
2. **Failed breakout**: price makes new high/low but closes inside prior range
3. **Wick rejection**: candle wick > `wick_ratio` of total range (optional)

### Signal Logic

```
volume_spike = volume > avg_volume * spike_mult
failed_up    = (high > prev_rolling_high) AND (close < prev_rolling_high)
failed_down  = (low < prev_rolling_low)   AND (close > prev_rolling_low)
upper_wick   = (high - body_top) / range > wick_ratio

SHORT entry: volume_spike AND failed_up AND (upper_wick OR relaxed)
LONG entry:  volume_spike AND failed_down AND (lower_wick OR relaxed)
```

### Usage

```python
from aqc.strategies.intraday import VolumeExhaustionStrategy

strategy = VolumeExhaustionStrategy(
    event_queue=eq,
    symbols=["AAPL"],
    volume_window=20,
    spike_mult=2.5,
    breakout_window=10,
    wick_ratio=0.6,
    require_wick_rejection=True,   # strict mode
    max_holding_bars=15,
    stop_loss_pct=0.015,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `volume_window` | int | 20 | Avg volume window |
| `spike_mult` | float | 2.5 | Volume spike multiplier |
| `breakout_window` | int | 10 | Range lookback for breakout detection |
| `wick_ratio` | float | 0.6 | Min wick/range ratio for rejection |
| `require_wick_rejection` | bool | True | Require all 3 conditions |
| `exit_volume_ratio` | float | 0.8 | Exit when vol drops below this x avg |
| `max_holding_bars` | int | 15 | Max holding period |
| `stop_loss_pct` | float | 0.015 | Stop-loss fraction |

---

## Strategy 3: Z-Score Mean Reversion (Adaptive)

### Alpha Thesis

A rolling z-score identifies statistical outliers, but fixed thresholds
fail in changing volatility regimes.  This strategy adapts its entry
thresholds based on the current vol regime:

```
adaptive_threshold = base_entry_z * (1 + vol_adjustment * (vol_ratio - 1))
```

In high-vol regimes, the threshold widens (requiring larger z-scores).
In low-vol regimes, the threshold tightens.

### Signal Computation

```
z = (close - rolling_mean) / rolling_std
vol_ratio = short_vol / long_vol
threshold = base_entry_z * (1 + vol_adjustment * (vol_ratio - 1))

Entry LONG:  z < -threshold
Entry SHORT: z > +threshold (if allow_short)
Exit:        |z| < base_exit_z
```

### Usage

```python
from aqc.strategies.intraday import ZScoreReversionStrategy

strategy = ZScoreReversionStrategy(
    event_queue=eq,
    symbols=["AAPL"],
    z_window=20,
    base_entry_z=2.0,
    base_exit_z=0.5,
    vol_lookback=60,
    vol_adjustment=0.5,
    allow_short=True,
    max_holding_bars=15,
    stop_loss_pct=0.02,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `z_window` | int | 20 | Rolling window for z-score |
| `base_entry_z` | float | 2.0 | Base z-score entry threshold |
| `base_exit_z` | float | 0.5 | Z-score exit threshold |
| `vol_lookback` | int | 60 | Long-term vol window |
| `vol_adjustment` | float | 0.5 | Threshold scaling factor |
| `allow_short` | bool | True | Enable short signals |
| `max_holding_bars` | int | 15 | Max holding period |
| `stop_loss_pct` | float | 0.02 | Stop-loss fraction |

---

## Strategy 4: Composite Alpha

### Alpha Thesis

Individual mean-reversion signals are noisy.  Combining orthogonal signals
into a composite alpha improves Sharpe:

```
alpha = w_vwap * vwap_signal + w_volume * volume_signal + w_zscore * z_signal
```

Each sub-signal is normalised to `[-1, 1]`:
- **Positive** = bullish (price extended down, expect reversion up)
- **Negative** = bearish (price extended up, expect reversion down)

A **minimum signal agreement** filter requires at least `min_signals`
sub-signals to agree in direction before entry.

### Usage

```python
from aqc.strategies.intraday import CompositeMeanReversionStrategy

strategy = CompositeMeanReversionStrategy(
    event_queue=eq,
    symbols=["AAPL"],
    w_vwap=0.4,
    w_volume=0.3,
    w_zscore=0.3,
    composite_threshold=0.3,   # min |alpha| to enter
    exit_threshold=0.1,        # alpha threshold to exit
    min_signals=2,             # at least 2 signals must agree
    max_holding_bars=20,
    stop_loss_pct=0.02,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `w_vwap` | float | 0.4 | Weight for VWAP signal |
| `w_volume` | float | 0.3 | Weight for volume signal |
| `w_zscore` | float | 0.3 | Weight for z-score signal |
| `composite_threshold` | float | 0.3 | Min alpha for entry |
| `exit_threshold` | float | 0.1 | Alpha threshold to exit |
| `min_signals` | int | 2 | Min agreeing sub-signals |
| `max_holding_bars` | int | 20 | Max holding period |
| `stop_loss_pct` | float | 0.02 | Stop-loss fraction |

---

## Walk-Forward Optimisation

All strategies are WFO-compatible because they follow the standard
`BaseStrategy(event_queue, symbols, **params)` constructor pattern:

```python
from aqc.research import (
    WalkForwardEngine, WalkForwardMode,
    ParameterSpace, IntParam, FloatParam,
    ObjectiveMetric,
)
from aqc.strategies.intraday import VWAPReversionStrategy

space = ParameterSpace()
space.add(IntParam("rolling_window", 10, 30, step=5))
space.add(FloatParam("entry_threshold", 1.5, 3.0, n_points=4))
space.add(FloatParam("exit_threshold", 0.2, 0.8, n_points=3))

engine = WalkForwardEngine(
    data=data,
    strategy_factory=VWAPReversionStrategy,
    parameter_space=space,
    mode=WalkForwardMode.ROLLING,
    train_period=120,
    test_period=60,
    optimizer="grid",
    objective=ObjectiveMetric.SHARPE,
)
result = engine.run()
```

---

## Research Analysis

### Signal Statistics

Extract signal-level data from trade logs:

```python
import pandas as pd

# After backtest
trade_df = pd.DataFrame(result["trade_log"])

# Per-signal-type breakdown
if "metadata" in trade_df.columns:
    trade_df["signal_type"] = trade_df["metadata"].apply(
        lambda m: m.get("signal_type", "unknown") if isinstance(m, dict) else "unknown"
    )
    print(trade_df.groupby("signal_type").agg({
        "realised_pnl": ["count", "mean", "std", "sum"],
    }))
```

### Alpha Decay Analysis

Measure how quickly the composite alpha decays after signal:

```python
# Collect alpha at signal time and subsequent returns
signal_bars = []  # track bars where signal fired
for i, row in enumerate(trade_log):
    if row.get("metadata", {}).get("action") == "entry":
        alpha_at_entry = row["metadata"]["composite_alpha"]
        # Track forward returns for next N bars
        ...
```

### Return Distributions

```python
import matplotlib.pyplot as plt

returns = trade_df["realised_pnl"].dropna()
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(returns, bins=30, color="#4FC3F7", edgecolor="#21262d", alpha=0.8)
ax.axvline(returns.mean(), color="white", linestyle="--")
ax.set_title("Trade Return Distribution")
plt.show()
```
