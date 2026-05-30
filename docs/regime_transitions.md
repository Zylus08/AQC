# Regime Transition Alpha Research

The Regime Transition Alpha Framework (Part 2 of Research Diagnostics) is designed to determine whether regime shifts themselves constitute exploitable alpha opportunities.

It answers:
- Do regime transitions (e.g., NORMAL -> HIGH volatility) predict future returns?
- Are these returns statistically significant?
- How frequently do specific regime transitions occur?

## Architecture

1. **TransitionEngine** (`aqc.research.regime_transitions.transition_engine`): Scans the historical regime timeline and records discrete `TransitionEvent` objects whenever a volatility or trend regime changes.
2. **TransitionAlphaAnalyzer** (`aqc.research.regime_transitions.transition_alpha`): Calculates forward returns (e.g., 1d, 3d, 5d, 10d, 20d) post-transition. It uses a two-sample t-test (Welch's t-test) to compare post-transition returns against unconditional market returns to evaluate statistical significance.
3. **TransitionReportGenerator**: Generates a tabular text report highlighting significant alpha.
4. **TransitionVisualizer**: Plots transition frequency matrices and return heatmaps.

## Usage

```python
from aqc.research.regime_transitions import (
    TransitionEngine,
    TransitionAlphaAnalyzer,
    TransitionReportGenerator,
    TransitionVisualizer
)

# 1. Identify Transitions
engine = TransitionEngine(regime_df, prices)
events_df = engine.get_events_df()

# 2. Analyze Forward Alpha
analyzer = TransitionAlphaAnalyzer(events_df, prices, horizons=[1, 3, 5, 10, 20])
alpha_df = analyzer.analyze_alpha()

# 3. Reports & Plots
t_rep = TransitionReportGenerator(alpha_df, horizons=[1, 3, 5, 10, 20])
t_rep.print_report()
t_rep.save_csv("reports/")

vis = TransitionVisualizer(analyzer)
vis.plot_all("reports/plots/")
```

## Generated Outputs

| File | Description |
|------|-------------|
| `reports/transition_alpha_report.csv` | Full alpha and t-statistics per transition. |
| `reports/transition_significance.csv` | p-values for all transitions across time horizons. |
| `reports/plots/transition_frequency_matrix.png` | Heatmap counting how often State A transitions to State B. |
| `reports/plots/transition_return_heatmap_vol.png` | Heatmap of forward returns for volatility transitions with significance markers (*). |
