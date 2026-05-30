# Capacity Analysis Framework

The Capacity Analysis Framework (Part 3 of Research Diagnostics) simulates how trading costs, slippage, and market impact affect a strategy as Assets Under Management (AUM) grow.

It answers:
- Does the strategy's alpha survive institutional-level capital?
- At what AUM does the Sharpe ratio break down?
- What is the maximum deployable capital before execution costs destroy profitability?

## Architecture

1. **SlippageModel** (`aqc.execution.slippage_model`): Models fixed and volatility-adjusted slippage.
2. **LiquidityModel** (`aqc.execution.liquidity_model`): Ensures the strategy respects maximum volume participation constraints based on ADV.
3. **SquareRootImpactModel** (`aqc.execution.market_impact`): Uses the standard institutional square-root model of market impact:
   `Impact = c * daily_volatility * sqrt(order_qty / ADV)`
4. **CapacityAnalyzer** (`aqc.execution.capacity_analyzer`): Orchestrates backtests across predefined capital tiers, injecting the execution constraints, and identifies the exact "capacity breakpoint" where Sharpe drops below a threshold.
5. **CapacityReportGenerator**: Generates cost curves and alpha decay frontiers.

## Usage

```python
from aqc.execution.capacity_analyzer import CapacityAnalyzer, CapacityConfig
from aqc.execution.capacity_reports import CapacityReportGenerator

# Define the function that runs your backtest and returns metrics
def run_backtest(capital, slippage_model, impact_model, liquidity_model):
    # run engine with capital
    # ...
    return {
        "sharpe": 2.1,
        "cagr": 0.15,
        "mdd": 0.10,
        "total_execution_cost": 50000.0
    }

# Define test scenarios
config = CapacityConfig(
    capital_levels=[1e6, 1e7, 5e7, 1e8, 5e8, 1e9],
    fixed_slippage_bps=2.0,
    impact_coefficient=0.1
)

analyzer = CapacityAnalyzer(run_backtest, config)

# Run capacity frontier
capacity_df = analyzer.run_capacity_analysis()

# Find the point where Sharpe decays by 50%
max_capital = analyzer.detect_capacity_breakpoint(sharpe_decay_threshold=0.5)

# Generate Reports
report_gen = CapacityReportGenerator(capacity_df, max_capital)
report_gen.print_report()
report_gen.save_csv("reports/")
report_gen.plot_all("reports/plots/")
```

## Generated Outputs

| File | Description |
|------|-------------|
| `reports/capacity_analysis_report.csv` | Sharpe, CAGR, MDD, and Execution Cost per capital level. |
| `reports/plots/sharpe_vs_capital.png` | Alpha Decay curve showing Sharpe dropping as AUM grows. |
| `reports/plots/cagr_vs_capital.png` | Capacity Frontier showing net returns vs capital. |
| `reports/plots/execution_cost_curve.png` | Execution cost (in bps of AUM) scaling with capital size. |
