# Volatility Forecasting Framework

## Overview

The AQC Volatility Forecasting Framework provides production-grade volatility
estimation, forecasting, regime detection, and position sizing.

```
aqc/volatility/
├── __init__.py                  # Package exports
├── ewma.py                     # EWMA volatility (RiskMetrics)
├── garch.py                    # GARCH(1,1) with MLE fitting
├── forecasting_engine.py       # Multi-model ensemble + regime detection
└── volatility_metrics.py       # Position sizing + metric utilities
```

---

## Module 1: EWMA Volatility (`ewma.py`)

### Theory

The EWMA model applies exponentially decaying weights to squared returns:

```
σ²_t = λ · σ²_{t-1} + (1 - λ) · r²_{t-1}
```

- `λ = 0.94` (RiskMetrics standard for daily data)
- No mean-reversion — equivalent to IGARCH(1,1)

### Functions

| Function | Description | Key Parameters |
|----------|-------------|----------------|
| `ewma_variance()` | Raw EWMA variance | `decay`, `min_periods` |
| `ewma_volatility()` | Annualised EWMA vol | `decay`, `annualise`, `ann_factor` |
| `ewma_forecast()` | h-step-ahead forecast | `decay`, `horizon` |

### Usage

```python
from aqc.volatility import ewma_volatility

vol = ewma_volatility(log_returns, decay=0.94, annualise=True)
```

---

## Module 2: GARCH(1,1) (`garch.py`)

### Theory

Bollerslev's GARCH(1,1):

```
σ²_t = ω + α · r²_{t-1} + β · σ²_{t-1}
```

- `ω > 0` constant (drives long-run variance)
- `α ≥ 0` ARCH coefficient (shock reaction)
- `β ≥ 0` GARCH coefficient (persistence)
- `α + β < 1` for stationarity

Long-run variance: `σ²_LR = ω / (1 - α - β)`

### Classes

| Class | Description |
|-------|-------------|
| `GARCH11` | MLE fitting + forecasting |
| `GARCHResult` | Fitted parameters + diagnostics |

### Usage

```python
from aqc.volatility import GARCH11

model = GARCH11(ann_factor=252)
result = model.fit(log_returns)

print(f"Alpha: {result.alpha:.4f}")
print(f"Beta:  {result.beta:.4f}")
print(f"Persistence: {result.persistence:.4f}")
print(f"LR Vol: {result.long_run_volatility:.2%}")
print(f"Half-life: {result.half_life:.1f} days")

# Forecast
fc = model.forecast(result, log_returns, horizon=5)
print(f"5d Forecast: {fc['forecast_vol_annualised']:.2%}")
print(f"CI 95%: [{fc['ci_lower_95']:.2%}, {fc['ci_upper_95']:.2%}]")

# Conditional volatility series
vol_series = model.conditional_volatility(result)
```

---

## Module 3: Forecasting Engine (`forecasting_engine.py`)

### Architecture

The engine combines three models into a weighted ensemble:

```
ensemble_vol = w_ewma × EWMA + w_garch × GARCH + w_hist × Historical
```

Default weights: `(0.40, 0.35, 0.25)`.

### Regime Detection

Classifies current vol into regimes using running percentiles:

| Regime | Condition |
|--------|-----------|
| LOW | vol ≤ 25th percentile |
| NORMAL | 25th < vol < 75th percentile |
| HIGH | 75th ≤ vol < 95th percentile |
| EXTREME | vol ≥ 95th percentile |

### Usage

```python
from aqc.volatility import VolatilityForecastEngine

engine = VolatilityForecastEngine(
    ewma_decay=0.94,
    hist_window=21,
    garch_refit_every=60,
    weights=(0.4, 0.35, 0.25),
)

# Single forecast
result = engine.fit_and_forecast(close_prices)
print(f"Forecast: {result.forecast_vol:.2%}")
print(f"CI: [{result.ci_lower:.2%}, {result.ci_upper:.2%}]")
print(f"Regime: {result.regime.value}")

# Full time series
df = engine.compute_full_series(close_prices)

# Generate CSV report
report = engine.generate_report(close_prices, "reports/volatility_report.csv")

# Plots
engine.plot_forecast_vs_realized(report)
engine.plot_volatility_clusters(report)
```

---

## Module 4: Position Sizing (`volatility_metrics.py`)

### Sizing Methods

#### 1. Volatility Targeting

```
Position Size = Dollar Risk / (Daily Vol × Price)
Daily Vol = Forecast Vol / √252
Dollar Risk = Equity × Risk Fraction
```

#### 2. Inverse Volatility

Weight each asset inversely proportional to its volatility:

```
weight_i = (1/vol_i) / Σ(1/vol_j)
```

#### 3. Risk Parity

Equal risk contribution from each position:

```
Target Risk Each = (Equity × Target Vol / √252) / N
Position = Target Risk / (Daily Vol × Price)
```

### Usage

```python
from aqc.volatility import VolatilitySizer, SizingMethod

sizer = VolatilitySizer(
    target_vol=0.10,        # 10% target portfolio vol
    risk_fraction=0.01,     # 1% of equity per trade
    max_position_pct=0.20,  # max 20% of equity per position
)

# Single position
result = sizer.size_position(
    "AAPL", price=180.0, forecast_vol=0.25, equity=100_000,
    method=SizingMethod.VOL_TARGET,
)
print(f"Quantity: {result.quantity}")
print(f"Weight: {result.weight:.2%}")

# Multi-asset portfolio
portfolio = sizer.size_portfolio(
    symbols=["AAPL", "MSFT", "GLD"],
    prices={"AAPL": 180.0, "MSFT": 420.0, "GLD": 195.0},
    vols={"AAPL": 0.28, "MSFT": 0.22, "GLD": 0.12},
    equity=500_000,
    method=SizingMethod.INVERSE_VOL,
)
```

### Portfolio Integration

The sizer integrates with the existing `Portfolio._size_order()` method.
Override it to use vol-targeting:

```python
class VolTargetPortfolio(Portfolio):
    def __init__(self, *args, sizer: VolatilitySizer, vol_engine, **kwargs):
        super().__init__(*args, **kwargs)
        self.sizer = sizer
        self.vol_engine = vol_engine
    
    def _size_order(self, signal):
        forecast = self.vol_engine.fit_and_forecast(self.prices[signal.symbol])
        result = self.sizer.size_position(
            signal.symbol, signal.suggested_price,
            forecast.forecast_vol, self.equity,
        )
        return result.quantity
```

---

## Metric Functions

| Function | Description | Returns |
|----------|-------------|---------|
| `volatility_cone()` | Realized vol percentiles across horizons | DataFrame |
| `vol_of_vol()` | Volatility-of-volatility metric | Series |
| `forecast_error_stats()` | MAE, RMSE, bias, correlation | dict |

### Volatility Cone

```python
from aqc.volatility import volatility_cone

cone = volatility_cone(
    prices,
    horizons=[5, 10, 21, 63, 126, 252],
    percentiles=[10, 25, 50, 75, 90],
)
# Returns: DataFrame with percentile columns + "current"
```

---

## Research Demo

```bash
python examples/run_volatility_research.py
```

Outputs:
- `reports/volatility_report.csv` — full time series
- `reports/vol_forecast_vs_realized.png`
- `reports/vol_clusters.png`
- `reports/vol_cone.png`
- `reports/vol_regime_changes.png`
