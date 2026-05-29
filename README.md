# AQC — AlgoQuant Club Quantitative Research Framework

> A modular, event-driven backtesting and quantitative research infrastructure
> designed for professional-quality strategy development.

---

## Architecture Overview

```
                        ┌─────────────────────────────────────────────┐
                        │              BacktestEngine                   │
                        │  (bar-by-bar event loop, no look-ahead bias) │
                        └────────────────────┬────────────────────────┘
                                             │  emits
                                             ▼
┌────────────┐  MarketEvent   ┌────────────────────┐  SignalEvent  ┌────────────────────┐
│ CSV/Parquet│ ─────────────► │     Strategy        │ ────────────► │      Portfolio      │
│  DataLoader│                │ (BaseStrategy +     │               │  (Position, PnL,    │
└────────────┘                │  generate_signal)   │               │   cash accounting)  │
                              └────────────────────┘               └────────┬───────────┘
                                                                            │  OrderEvent (after RiskManager)
                                                                            ▼
                                                                   ┌────────────────────┐
                                                                   │  ExecutionEngine    │
                                                                   │  (SimulatedBroker,  │
                                                                   │   commission +      │
                                                                   │   slippage models)  │
                                                                   └────────┬───────────┘
                                                                            │  FillEvent
                                                                            ▼
                                                                   ┌────────────────────┐
                                                                   │  Portfolio update   │
                                                                   │  + equity curve     │
                                                                   └────────────────────┘
                                                                            │
                                                                            ▼
                                                                   ┌────────────────────┐
                                                                   │  PerformanceMetrics │
                                                                   │  + ReportGenerator  │
                                                                   └────────────────────┘
```

### Event Flow

```
MarketEvent → Strategy → SignalEvent → Portfolio → RiskManager → OrderEvent → Broker → FillEvent → Portfolio Update
```

---

## Project Structure

```
aqc/
├── data/
│   ├── loaders/
│   │   └── csv_loader.py         # CSV OHLCV loader with schema validation
│   ├── preprocess/
│   │   └── cleaner.py            # Data cleaning pipeline
│   └── storage/
│       └── parquet_store.py      # Fast Parquet cache
│
├── backtester/
│   ├── event.py                  # MarketEvent, SignalEvent, OrderEvent, FillEvent
│   ├── event_queue.py            # Thread-safe event queue
│   ├── portfolio.py              # Position accounting + PnL
│   ├── broker.py                 # Commission/slippage models + SimulatedBroker
│   ├── execution.py              # ExecutionEngine facade
│   └── engine.py                 # BacktestEngine (main event loop)
│
├── research/                       # Walk-Forward Optimisation (NEW)
│   ├── parameter_space.py        # IntParam, FloatParam, CategoricalParam, ParameterGrid
│   ├── optimizer.py              # GridSearchOptimizer, RandomSearchOptimizer
│   ├── walk_forward.py           # WalkForwardEngine (rolling/expanding windows)
│   └── validation.py             # IS/OOS statistics, plots, reports
│
├── strategies/
│   ├── base_strategy.py          # Abstract BaseStrategy
│   └── sample_strategy.py        # SMACrossover, RSIMeanReversion, EMAMomentum
│
├── indicators/
│   ├── moving_averages.py        # SMA, EMA, WMA, DEMA, HMA
│   ├── momentum.py               # RSI, MACD, Stochastic, ROC
│   └── volatility.py             # Bollinger Bands, ATR, Historical Vol
│
├── risk/
│   └── risk_manager.py           # RiskManager with 4 rule checks
│
├── analytics/
│   ├── metrics.py                # Sharpe, Sortino, CAGR, MDD, Win Rate, etc.
│   └── reporting.py              # Console + CSV report generation
│
├── utils/
│   ├── logger.py                 # Rotating file + console logging
│   └── config_loader.py          # YAML config with deep-merge defaults
│
configs/
│   └── config.yaml               # Full YAML configuration
│
tests/
│   ├── test_events.py
│   ├── test_portfolio.py
│   ├── test_risk.py
│   ├── test_metrics.py
│   ├── test_integration.py
│   └── test_wfo.py               # Walk-forward optimisation tests (38 tests)
│
examples/
│   └── run_walk_forward.py       # WFO demo with synthetic data
│
docs/                             # Extended documentation
main.py                           # Entry point
requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the backtest (with synthetic data)

If no CSV file exists in `data/raw/`, the engine automatically generates
synthetic OHLCV data so you can verify the pipeline immediately:

```bash
python main.py
```

### 3. Run with a specific strategy

```bash
python main.py --strategy rsi_mean_reversion
python main.py --strategy ema_momentum --capital 500000
```

### 4. Run with your own CSV data

Place an OHLCV CSV file in `data/raw/`:

```
data/raw/AAPL.csv
```

Supported CSV format:
```csv
date,open,high,low,close,volume
2024-01-02,150.05,152.30,149.80,151.90,45000000
```

Then run:
```bash
python main.py --symbol AAPL
```

### 5. Run Walk-Forward Optimisation

```bash
python examples/run_walk_forward.py
```

See [docs/walk_forward_guide.md](docs/walk_forward_guide.md) for full WFO documentation.

### 6. Run tests

```bash
pytest                                    # 101 tests
pytest --cov=aqc --cov-report=term-missing
```

---

## Configuration

All parameters are controlled via `configs/config.yaml`.

Key sections:

| Section | Key Parameters |
|---------|---------------|
| `backtest` | `symbols`, `initial_capital`, `start_date`, `end_date` |
| `strategy` | `name` (sma_crossover / rsi_mean_reversion / ema_momentum), `params` |
| `broker` | `commission_model`, `commission_rate`, `slippage_bps` |
| `risk` | `max_position_pct_equity`, `max_daily_loss_pct`, `max_open_positions` |
| `logging` | `level`, `log_dir`, `log_to_file` |
| `output` | `reports_dir`, `export_equity_curve`, `export_trade_log` |

---

## Building a Custom Strategy

```python
from aqc.strategies.base_strategy import BaseStrategy
from aqc.backtester.event import SignalEvent, SignalDirection
from aqc.indicators.momentum import rsi
import pandas as pd
from typing import Optional

class MyStrategy(BaseStrategy):
    """My custom mean-reversion strategy."""

    @property
    def min_bars_required(self) -> int:
        return 30

    def generate_signal(self, symbol: str, bars: pd.DataFrame) -> Optional[SignalEvent]:
        rsi_val = rsi(bars["close"], 14).iloc[-1]

        if rsi_val < 30:
            return SignalEvent(
                symbol=symbol,
                strategy_id=self.strategy_id,
                direction=SignalDirection.LONG,
                strength=1.0,
            )
        elif rsi_val > 70:
            return SignalEvent(
                symbol=symbol,
                strategy_id=self.strategy_id,
                direction=SignalDirection.EXIT,
                strength=1.0,
            )
        return None
```

---

## Implemented Strategies

| Strategy | Description | Key Parameters |
|----------|-------------|----------------|
| `SMACrossoverStrategy` | Classic golden/death cross | `fast_period`, `slow_period` |
| `RSIMeanReversionStrategy` | RSI oversold/overbought | `rsi_period`, `oversold`, `overbought`, `allow_short` |
| `EMAMomentumStrategy` | EMA alignment with trend filter | `short_period`, `medium_period`, `long_period` |

---

## Performance Metrics

| Metric | Description |
|--------|-------------|
| Sharpe Ratio | Annualised risk-adjusted return (vs 4% Rf) |
| Sortino Ratio | Downside-only risk-adjusted return |
| Max Drawdown | Peak-to-trough equity decline |
| CAGR | Compound Annual Growth Rate |
| Calmar Ratio | CAGR / Max Drawdown |
| Win Rate | Fraction of profitable trades |
| Profit Factor | Gross profit / Gross loss |
| Avg Trade Return | Mean PnL per closed trade |
| Exposure | Fraction of time with open positions |

---

## Development Roadmap

### Phase 1 — Core Framework (✅ Complete)
- [x] Event-driven backtesting engine
- [x] CSV data loader with validation
- [x] Portfolio with average-cost accounting
- [x] Simulated broker (commission + slippage)
- [x] Risk manager (4 rule checks)
- [x] Performance metrics (12 metrics)
- [x] YAML configuration system
- [x] Logging (rotating file + console)
- [x] Unit tests (events, portfolio, risk, metrics, integration)

### Phase 2 — Research Infrastructure (✅ In Progress)
- [ ] Parquet data pipeline with caching
- [ ] Multi-symbol portfolio management
- [x] Walk-forward optimisation framework
- [x] Grid Search + Random Search optimisers
- [x] Parameter stability analysis (CV, heatmaps)
- [x] IS/OOS correlation & overfitting detection
- [x] Publication-quality validation plots
- [ ] Regime detection (Hidden Markov Models)

### Phase 3 — Advanced Strategies
- [ ] Statistical arbitrage (pairs trading)
- [ ] Order book imbalance signals
- [ ] Volatility forecasting (GARCH)
- [ ] Intraday mean reversion
- [ ] Reinforcement learning agents (Stable-Baselines3)

### Phase 4 — Live Trading
- [ ] Broker API adapters (Alpaca, Interactive Brokers)
- [ ] Real-time data feed integration
- [ ] Paper trading mode
- [ ] Execution risk controls (circuit breakers)

---

## Engineering Standards

- **Python 3.12** with full type hints
- **SOLID** principles throughout
- **Dataclasses** for immutable event types
- **ABC** for extensible interfaces (strategy, broker, commission, slippage)
- **Dependency injection** — all components receive collaborators via constructor
- **Zero look-ahead bias** — strategy only sees data up to the current bar
- **PEP 8** compliant
- **Comprehensive docstrings** on every public class and method

---

## License

MIT — AlgoQuant Club
