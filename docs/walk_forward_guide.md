# Walk-Forward Optimisation Guide

## What is Walk-Forward Testing?

Walk-forward optimisation (WFO) is the industry-standard method for validating
trading strategies without overfitting.  Instead of optimising parameters on the
full historical dataset, WFO:

1. Divides data into sequential train/test **folds**.
2. Optimises parameters on each **training window** (in-sample).
3. Evaluates the best parameters on the subsequent **test window** (out-of-sample).
4. Aggregates test results to estimate realistic live performance.

```
 Full dataset
 ┌─────────────────────────────────────────────────────────────────┐
 │ 2020        │ 2021        │ 2022        │ 2023                  │
 └─────────────────────────────────────────────────────────────────┘

 Fold 0:  [═══ TRAIN ═══]  [── TEST ──]
 Fold 1:        [═══ TRAIN ═══]  [── TEST ──]
 Fold 2:              [═══ TRAIN ═══]  [── TEST ──]
```

## Quick Start

```python
from aqc.research import (
    WalkForwardEngine, WalkForwardMode,
    ParameterSpace, IntParam,
    ObjectiveMetric, WalkForwardValidator,
)
from aqc.strategies.sample_strategy import SMACrossoverStrategy

# 1. Define parameter search space
space = ParameterSpace()
space.add(IntParam("fast_period", low=5, high=30, step=5))
space.add(IntParam("slow_period", low=20, high=100, step=10))

# 2. Build and run the WFO engine
engine = WalkForwardEngine(
    data={"AAPL": ohlcv_df},
    strategy_factory=SMACrossoverStrategy,
    parameter_space=space,
    mode=WalkForwardMode.ROLLING,
    train_period="365D",          # 1 year training window
    test_period="90D",            # 3 month test window
    optimizer="grid",             # "grid" or "random"
    objective=ObjectiveMetric.SHARPE,
    initial_capital=100_000,
)

result = engine.run()

# 3. Export results
result.save_csv("reports/walk_forward_results.csv")

# 4. Validate and visualise
validator = WalkForwardValidator(result=result, output_dir="reports")
validator.print_report()
validator.plot_equity_curves(save=True)
validator.plot_sharpe_distribution(save=True)
validator.plot_parameter_stability(save=True)
```

---

## Module Reference

### `aqc/research/parameter_space.py`

| Class | Description |
|-------|-------------|
| `IntParam(name, low, high, step)` | Integer parameter with step size |
| `FloatParam(name, low, high, n_points)` | Float parameter with n grid points |
| `CategoricalParam(name, choices)` | Discrete unordered choices |
| `ParameterSpace` | Container for all parameter dimensions |
| `ParameterGrid(space)` | Iterable Cartesian product |

```python
space = ParameterSpace()
space.add(IntParam("rsi_period", 10, 20, step=2))
space.add(FloatParam("oversold", 20.0, 40.0, n_points=5))
space.add(CategoricalParam("allow_short", [True, False]))

print(space.grid_size())    # 6 × 5 × 2 = 60
print(space.sample())       # {"rsi_period": 14, "oversold": 30.0, "allow_short": False}
```

### `aqc/research/optimizer.py`

| Class | Description |
|-------|-------------|
| `GridSearchOptimizer` | Exhaustive Cartesian product search |
| `RandomSearchOptimizer` | Randomised sampling (efficient for large spaces) |
| `ObjectiveMetric` | Enum of metrics to maximise |
| `OptimizationResult` | Best params + full evaluation history |

**Objective Metrics:**

| Enum Value | Metric |
|-----------|--------|
| `SHARPE` | Annualised Sharpe Ratio |
| `SORTINO` | Annualised Sortino Ratio |
| `CALMAR` | Calmar Ratio (CAGR / Max DD) |
| `CAGR` | Compound Annual Growth Rate |
| `MAX_DRAWDOWN` | Maximum Drawdown (negated for maximisation) |
| `WIN_RATE` | Win Rate |
| `PROFIT_FACTOR` | Gross Profit / Gross Loss |
| `TOTAL_RETURN` | Total Return % |

### `aqc/research/walk_forward.py`

| Class | Description |
|-------|-------------|
| `WalkForwardEngine` | Main WFO orchestrator |
| `WalkForwardMode` | `ROLLING`, `EXPANDING`, `ANCHORED` |
| `WalkForwardFold` | Per-fold metadata and results |
| `WalkForwardResult` | Aggregated multi-fold results |

**WalkForwardEngine Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `dict[str, DataFrame]` | — | OHLCV data |
| `strategy_factory` | callable | — | `(event_queue, symbols, **params) -> Strategy` |
| `parameter_space` | `ParameterSpace` | — | Search space |
| `mode` | `WalkForwardMode` | `ROLLING` | Window mode |
| `train_period` | `str \| int` | `"365D"` | Training window |
| `test_period` | `str \| int` | `"90D"` | Test window |
| `n_folds` | `int \| None` | `None` | Max folds (auto if None) |
| `optimizer` | `str` | `"grid"` | `"grid"` or `"random"` |
| `n_iter` | `int` | `50` | Random search iterations |
| `objective` | `ObjectiveMetric` | `SHARPE` | Metric to maximise |
| `initial_capital` | `float` | `100_000` | Backtest capital |

### `aqc/research/validation.py`

| Method | Description |
|--------|-------------|
| `print_report()` | Print full validation report |
| `save_report(filename)` | Save report to text file |
| `is_oos_correlation()` | IS/OOS Pearson correlation |
| `overfitting_score()` | Overfitting coefficient [0, 1] |
| `plot_equity_curves()` | Per-fold + combined equity curve |
| `plot_sharpe_distribution()` | IS vs OOS Sharpe histogram |
| `plot_parameter_stability()` | Heatmap + trajectory per fold |
| `plot_all()` | All three plots |

---

## Window Modes

### Rolling (default)

The training window shifts forward by `test_period` each fold.  Best for
strategies where older data becomes stale.

```
train=120 bars, test=60 bars:

Fold 0: bars [0,   119]  test: [120, 179]
Fold 1: bars [60,  179]  test: [180, 239]
Fold 2: bars [120, 239]  test: [240, 299]
```

### Expanding

The training window always starts at bar 0 and expands each fold.  Suitable
for strategies that improve with more data.

```
train=120 bars, test=60 bars:

Fold 0: bars [0,   119]  test: [120, 179]
Fold 1: bars [0,   179]  test: [180, 239]
Fold 2: bars [0,   239]  test: [240, 299]
```

---

## Writing a Custom Strategy for WFO

The strategy factory must have this signature:

```python
def my_factory(event_queue, symbols: list[str], **params) -> BaseStrategy:
    return MyStrategy(event_queue=event_queue, symbols=symbols, **params)
```

Or simply use the strategy class directly if its `__init__` matches:

```python
class RSIMeanReversionStrategy(BaseStrategy):
    def __init__(self, event_queue, symbols, rsi_period=14, oversold=30, overbought=70, ...):
        ...

# Factory = class itself (Python calls __init__ with the kwargs)
engine = WalkForwardEngine(
    ...,
    strategy_factory=RSIMeanReversionStrategy,
    parameter_space=space,
)
```

---

## Interpreting Results

### IS/OOS Correlation

| Value | Interpretation |
|-------|---------------|
| > 0.6 | Strong — parameters generalise well |
| 0.2 – 0.6 | Moderate — some overfitting present |
| < 0.2 | Weak/negative — strategy likely overfit |

### Parameter Stability (CV = σ/μ)

| CV | Interpretation |
|----|---------------|
| < 0.1 | Very stable — strategy not sensitive to this parameter |
| 0.1 – 0.3 | Moderately stable |
| > 0.3 | Unstable — parameter changes significantly across folds |

### Overfitting Score

* `0.0` — no overfitting (OOS return = IS return)
* `0.5` — OOS return is 50% of IS return (moderate overfitting)
* `1.0` — OOS return is 0 while IS is positive (heavy overfitting)

---

## Outputs

| File | Description |
|------|-------------|
| `reports/walk_forward_results.csv` | Per-fold metrics, best params, dates |
| `reports/wfo_equity_curves.png` | Per-fold + combined OOS equity curves |
| `reports/wfo_sharpe_distribution.png` | IS vs OOS Sharpe histograms |
| `reports/wfo_parameter_stability.png` | Heatmap and parameter trajectory |
| `reports/walk_forward_report.txt` | Full text validation report |
