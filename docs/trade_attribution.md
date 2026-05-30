# Trade Attribution Framework

The Trade Attribution Framework (Part 1 of the Research Diagnostics) decomposes portfolio performance at the trade level to explain **where** returns come from and **why**.

It answers questions like:
- Which trades generated the most returns?
- Does the strategy work better in high volatility or low volatility?
- Which specific signals (e.g. VWAP vs Volume Exhaustion) are driving the alpha?

## Architecture

The framework consists of four primary components:

1. **TradeAnalyzer** (`aqc.diagnostics.trade_analyzer`): Reconstructs round-trip trades (LONG or SHORT) from the raw `fill_log` produced by the portfolio engine. It annotates each trade with holding duration, MFE (Maximum Favourable Excursion), MAE (Maximum Adverse Excursion), entry/exit volatility, and the active regime at entry.
2. **TradeAttributionEngine** (`aqc.diagnostics.trade_attribution`): Decomposes the annotated trades across multiple dimensions: volatility regime, trend regime, duration bucket, and signal source.
3. **TradeVisualizer** (`aqc.diagnostics.trade_visualization`): Generates heatmaps, histograms, and bar charts.
4. **TradeReportGenerator** (`aqc.diagnostics.trade_reports`): Exports formatted console reports and CSV dumps.

## Usage

```python
from aqc.diagnostics.trade_analyzer import TradeAnalyzer
from aqc.diagnostics.trade_attribution import TradeAttributionEngine
from aqc.diagnostics.trade_reports import TradeReportGenerator
from aqc.diagnostics.trade_visualization import TradeVisualizer

# 1. Reconstruct trades
analyzer = TradeAnalyzer(portfolio.trade_log, prices_df, regime_df, vol_series)
trades_df = analyzer.to_dataframe()

# 2. Decompose PnL
engine = TradeAttributionEngine(trades_df)
attribution_summary = engine.full_summary()

# 3. Generate Reports & Plots
report_gen = TradeReportGenerator(trades_df, attribution_summary, analyzer.trade_stats())
report_gen.print_report()
report_gen.save_all("reports/")

vis = TradeVisualizer(attribution_summary, trades_df)
vis.plot_all("reports/plots/")
```

## Generated Outputs

| File | Description |
|------|-------------|
| `reports/trade_attribution_report.csv` | Full annotated trade log. |
| `reports/top_winners.csv` | Top 10 winning trades by PnL. |
| `reports/top_losers.csv` | Top 10 losing trades by PnL. |
| `reports/by_vol_regime.csv` | PnL summary broken down by Volatility Regime. |
| `reports/by_signal.csv` | PnL summary broken down by Strategy/Signal. |
| `reports/plots/pnl_by_regime.png` | Bar chart of PnL across volatility regimes. |
| `reports/plots/regime_duration_heatmap.png` | Heatmap of PnL by regime and holding duration. |
| `reports/plots/mfe_mae_scatter.png` | Scatter plot of Trade Runup (MFE) vs Drawdown (MAE). |
