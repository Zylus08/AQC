# AQC Architecture Guide

## System Design

### Event-Driven Paradigm

The AQC engine is built on a strict event-driven architecture.  No module
calls another module's methods directly to trigger work.  Instead, every
action produces an event that is placed on the shared :class:`EventQueue`.

This design yields several benefits:

* **Zero look-ahead bias**: strategies only process events for the current bar.
* **Modularity**: each component can be replaced without touching others.
* **Testability**: events can be injected directly into any component in tests.
* **Future extensibility**: a live trading broker simply consumes the same
  `OrderEvent` and emits a `FillEvent` — no engine changes required.

### Event Lifecycle

```
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│  Market  │────►│  Signal  │────►│  Order   │────►│   Fill   │
│  Event   │     │  Event   │     │  Event   │     │  Event   │
└──────────┘     └──────────┘     └──────────┘     └──────────┘
    │                 │                │                │
  Engine           Strategy        Portfolio         Broker
  emits            emits           emits             emits
```

### Component Responsibilities

| Component | Input | Output | Responsibility |
|-----------|-------|--------|----------------|
| `BacktestEngine` | Data dict | Results dict | Event loop orchestration |
| `BaseStrategy` | `MarketEvent` | `SignalEvent` | Alpha signal generation |
| `Portfolio` | `SignalEvent`, `FillEvent` | `OrderEvent` | Position accounting |
| `RiskManager` | `OrderEvent` | Approval/rejection | Risk gate |
| `ExecutionEngine` | `OrderEvent` | `FillEvent` | Order routing |
| `SimulatedBroker` | `OrderEvent` | `FillEvent` | Fill simulation |

## Data Flow Detail

```
1. BacktestEngine iterates over the sorted time index (bar-by-bar)

2. For each bar:
   a. Emit MarketEvent → ExecutionEngine price cache
   b. Emit MarketEvent → Portfolio.on_market_event() (mark-to-market)
   c. Emit MarketEvent → Strategy.on_market_event()
      → Strategy calls generate_signal()
      → If signal: puts SignalEvent on EventQueue

3. Drain EventQueue until empty:
   SIGNAL  → Portfolio.on_signal_event()
              → RiskManager.validate_order()
              → If approved: puts OrderEvent on EventQueue
   ORDER   → ExecutionEngine.on_order_event()
              → SimulatedBroker.execute_order()
              → Puts FillEvent on EventQueue
   FILL    → Portfolio.on_fill_event()
              → Updates cash, position, trade log

4. After all bars: compile results → PerformanceMetrics → ReportGenerator
```

## Adding a New Strategy

1. Create `aqc/strategies/my_strategy.py`
2. Inherit from `BaseStrategy`
3. Implement `generate_signal(symbol, bars) -> Optional[SignalEvent]`
4. Set `min_bars_required` property
5. Register in `main.py`'s `build_strategy()` factory

## Adding a New Commission Model

1. Subclass `CommissionModel` in `aqc/backtester/broker.py`
2. Implement `calculate(order, fill_price) -> float`
3. Pass to `SimulatedBroker(commission_model=MyModel())`

## Adding a New Risk Rule

1. Subclass `RiskManager`
2. Override `_custom_rules(order, portfolio) -> tuple[bool, str]`
3. Pass your subclass to `Portfolio(risk_manager=MyRiskManager())`
