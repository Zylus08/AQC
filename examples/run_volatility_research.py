"""
examples/run_volatility_research.py
=====================================
Volatility Forecasting Research Report Generator.

Generates:
- EWMA / GARCH / Historical volatility comparison
- Forecast vs realized plots
- Volatility clustering analysis
- Regime change timeline
- Vol cone analysis
- Position sizing comparison
- volatility_report.csv

Usage::

    python examples/run_volatility_research.py

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from aqc.utils.logger import setup_logging

setup_logging(level="WARNING", log_dir="logs", log_to_file=True)
logger = logging.getLogger("aqc.examples.volatility")
logger.setLevel(logging.INFO)

from aqc.volatility import (
    ewma_volatility,
    GARCH11,
    VolatilityForecastEngine,
    VolatilitySizer,
    SizingMethod,
    volatility_cone,
    vol_of_vol,
    forecast_error_stats,
)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def generate_regime_data(n: int = 1000, seed: int = 42) -> pd.Series:
    """Generate price data with distinct volatility regimes.

    Creates 4 regimes:
    - Bars   0-250: Low vol (sigma=0.005)
    - Bars 250-500: Normal vol (sigma=0.012)
    - Bars 500-750: High vol (crisis, sigma=0.025)
    - Bars 750-1000: Return to normal (sigma=0.010)
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n)

    regimes = [
        (0, 250, 0.005, "Low Vol"),
        (250, 500, 0.012, "Normal"),
        (500, 750, 0.025, "Crisis"),
        (750, n, 0.010, "Recovery"),
    ]

    returns = np.zeros(n)
    for start, end, sigma, _ in regimes:
        returns[start:end] = rng.normal(0.0002, sigma, end - start)

    prices = np.exp(np.cumsum(returns)) * 100.0
    return pd.Series(
        prices,
        index=pd.DatetimeIndex(dates, name="timestamp"),
        name="close",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    import math

    sep = "=" * 70
    thin = "-" * 70

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    print(f"\n{sep}")
    print("  AQC VOLATILITY FORECASTING RESEARCH REPORT")
    print(sep)

    # Generate data
    prices = generate_regime_data(1000, seed=42)
    log_ret = np.log(prices / prices.shift(1)).dropna()
    n = len(prices)
    print(f"\n  Data: {n} bars with 4 volatility regimes")
    print(f"  Date range: {prices.index[0].date()} to {prices.index[-1].date()}")

    # ------------------------------------------------------------------
    # 1. GARCH(1,1) Fit
    # ------------------------------------------------------------------
    print(f"\n{thin}")
    print("  1. GARCH(1,1) MODEL FIT")
    print(thin)

    model = GARCH11()
    result = model.fit(log_ret)
    print(f"\n  Converged:     {result.converged}")
    print(f"  omega:         {result.omega:.8f}")
    print(f"  alpha:         {result.alpha:.4f}")
    print(f"  beta:          {result.beta:.4f}")
    print(f"  Persistence:   {result.persistence:.4f}")
    print(f"  Half-life:     {result.half_life:.1f} days")
    print(f"  LR Volatility: {result.long_run_volatility*100:.2f}%")
    print(f"  Log-likelihood: {result.log_likelihood:.2f}")

    # Forecast
    fc = model.forecast(result, log_ret, horizon=5)
    print(f"\n  5-day Forecast:")
    print(f"    Vol (ann):  {fc['forecast_vol_annualised']*100:.2f}%")
    print(f"    CI 95%:     [{fc['ci_lower_95']*100:.2f}%, {fc['ci_upper_95']*100:.2f}%]")

    # ------------------------------------------------------------------
    # 2. Multi-Model Ensemble
    # ------------------------------------------------------------------
    print(f"\n{thin}")
    print("  2. MULTI-MODEL ENSEMBLE FORECAST")
    print(thin)

    engine = VolatilityForecastEngine(
        ewma_decay=0.94,
        hist_window=21,
        garch_refit_every=60,
        weights=(0.4, 0.35, 0.25),
    )
    forecast = engine.fit_and_forecast(prices)

    print(f"\n  Ensemble Forecast: {forecast.forecast_vol*100:.2f}%")
    print(f"  CI 95%:            [{forecast.ci_lower*100:.2f}%, {forecast.ci_upper*100:.2f}%]")
    print(f"  Regime:            {forecast.regime.value}")
    print(f"\n  Model breakdown:")
    for model_name, vol in forecast.model_vols.items():
        print(f"    {model_name:>12}: {vol*100:.2f}%")

    # ------------------------------------------------------------------
    # 3. Full Time Series Analysis
    # ------------------------------------------------------------------
    print(f"\n{thin}")
    print("  3. VOLATILITY TIME SERIES")
    print(thin)

    vol_df = engine.generate_report(prices, output_path=str(reports_dir / "volatility_report.csv"))
    print(f"\n  Saved: reports/volatility_report.csv ({len(vol_df)} rows)")

    # Regime distribution
    regime_counts = vol_df["regime"].value_counts()
    print(f"\n  Regime Distribution:")
    for regime, count in regime_counts.items():
        pct = count / len(vol_df) * 100
        print(f"    {regime:>8}: {count:>4} bars ({pct:.1f}%)")

    # ------------------------------------------------------------------
    # 4. Volatility Cone
    # ------------------------------------------------------------------
    print(f"\n{thin}")
    print("  4. VOLATILITY CONE")
    print(thin)

    cone = volatility_cone(prices)
    print(f"\n  {'Horizon':>8} {'P10':>8} {'P25':>8} {'P50':>8} {'P75':>8} {'P90':>8} {'Current':>8}")
    print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for h, row in cone.iterrows():
        print(
            f"  {h:>8} "
            f"{row['p10']*100:>7.2f}% "
            f"{row['p25']*100:>7.2f}% "
            f"{row['p50']*100:>7.2f}% "
            f"{row['p75']*100:>7.2f}% "
            f"{row['p90']*100:>7.2f}% "
            f"{row['current']*100:>7.2f}%"
        )

    # ------------------------------------------------------------------
    # 5. Forecast Error Analysis
    # ------------------------------------------------------------------
    print(f"\n{thin}")
    print("  5. FORECAST ERROR ANALYSIS")
    print(thin)

    if "realized_1d" in vol_df.columns:
        stats = forecast_error_stats(vol_df["ensemble_vol"], vol_df["realized_1d"])
        print(f"\n  MAE:          {stats['mae']*100:.4f}%")
        print(f"  RMSE:         {stats['rmse']*100:.4f}%")
        print(f"  Bias:         {stats['bias']*100:.4f}%")
        print(f"  Correlation:  {stats['correlation']:.4f}")
        print(f"  Hit Rate:     {stats['hit_rate']*100:.1f}%")

    # ------------------------------------------------------------------
    # 6. Position Sizing Demo
    # ------------------------------------------------------------------
    print(f"\n{thin}")
    print("  6. VOLATILITY-TARGETED POSITION SIZING")
    print(thin)

    sizer = VolatilitySizer(target_vol=0.10, risk_fraction=0.01)
    last_price = float(prices.iloc[-1])
    equity = 500_000

    methods = [SizingMethod.VOL_TARGET, SizingMethod.INVERSE_VOL, SizingMethod.RISK_PARITY]
    print(f"\n  Price: ${last_price:.2f}  |  Forecast Vol: {forecast.forecast_vol*100:.2f}%  |  Equity: ${equity:,.0f}")
    print(f"\n  {'Method':<18} {'Qty':>6} {'Weight':>8} {'$Risk':>10}")
    print(f"  {'-'*18} {'-'*6} {'-'*8} {'-'*10}")

    for method in methods:
        r = sizer.size_position(
            "SPY", last_price, forecast.forecast_vol, equity, method,
        )
        print(f"  {method.value:<18} {r.quantity:>6} {r.weight*100:>7.2f}% ${r.dollar_risk:>9.2f}")

    # Multi-asset portfolio
    print(f"\n  Multi-Asset Portfolio (Inverse Vol):")
    multi_vols = {"AAPL": 0.28, "MSFT": 0.22, "JNJ": 0.14, "GLD": 0.12, "TLT": 0.10}
    multi_prices = {"AAPL": 180.0, "MSFT": 420.0, "JNJ": 155.0, "GLD": 195.0, "TLT": 92.0}
    portfolio_sizing = sizer.size_portfolio(
        list(multi_vols.keys()), multi_prices, multi_vols, equity, SizingMethod.INVERSE_VOL,
    )
    print(f"\n  {'Symbol':<8} {'Qty':>6} {'Weight':>8} {'Vol':>8} {'$Risk':>10}")
    print(f"  {'-'*8} {'-'*6} {'-'*8} {'-'*8} {'-'*10}")
    for s, r in sorted(portfolio_sizing.items(), key=lambda x: -x[1].weight):
        print(f"  {s:<8} {r.quantity:>6} {r.weight*100:>7.2f}% {r.forecast_vol*100:>7.1f}% ${r.dollar_risk:>9.2f}")

    # ------------------------------------------------------------------
    # 7. Generate Plots
    # ------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Plot 1: Forecast vs Realized
        engine.plot_forecast_vs_realized(vol_df, save=True, output_dir=str(reports_dir))

        # Plot 2: Volatility Clusters
        engine.plot_volatility_clusters(vol_df, save=True, output_dir=str(reports_dir))

        # Plot 3: Volatility Cone
        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(12, 6), facecolor="#0d1117")
        ax.set_facecolor("#161b22")

        cone_filled = volatility_cone(prices, percentiles=[10, 25, 50, 75, 90])
        horizons_plot = cone_filled.index.values
        ax.fill_between(horizons_plot, cone_filled["p10"] * 100, cone_filled["p90"] * 100,
                         alpha=0.15, color="#4FC3F7", label="10-90th %ile")
        ax.fill_between(horizons_plot, cone_filled["p25"] * 100, cone_filled["p75"] * 100,
                         alpha=0.25, color="#4FC3F7", label="25-75th %ile")
        ax.plot(horizons_plot, cone_filled["p50"] * 100, color="#FFB74D",
                linewidth=2, label="Median", marker="o")
        ax.plot(horizons_plot, cone_filled["current"] * 100, color="#FF5252",
                linewidth=2, linestyle="--", label="Current", marker="s")
        ax.set_xlabel("Horizon (days)", color="white", fontsize=11)
        ax.set_ylabel("Annualised Volatility (%)", color="white", fontsize=11)
        ax.set_title("Volatility Cone", color="white", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10, facecolor="#21262d", labelcolor="white")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        plt.tight_layout()
        fig.savefig(reports_dir / "vol_cone.png", dpi=150,
                    bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

        # Plot 4: Regime Changes
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(14, 8), facecolor="#0d1117",
            height_ratios=[2, 1], sharex=True,
        )
        for ax in (ax1, ax2):
            ax.set_facecolor("#161b22")

        # Price with regime coloring
        ax1.plot(prices.index, prices.values, color="#4FC3F7", linewidth=1.2, label="Price")
        ax1.set_ylabel("Price", color="white", fontsize=11)
        ax1.set_title("Price & Volatility Regime Changes",
                       color="white", fontsize=14, fontweight="bold")
        ax1.legend(fontsize=9, facecolor="#21262d", labelcolor="white")

        # Vol with regime shading
        vol_plot = vol_df["ensemble_vol"] * 100
        ax2.plot(vol_df.index, vol_plot, color="#FFB74D", linewidth=1.2)
        regime_colors = {"LOW": "#1B5E20", "NORMAL": "#1565C0", "HIGH": "#E65100", "EXTREME": "#B71C1C"}
        for regime, color in regime_colors.items():
            mask = vol_df["regime"] == regime
            if mask.any():
                ax2.fill_between(vol_df.index, 0, vol_plot.max(), where=mask,
                                  alpha=0.15, color=color, label=regime)
        ax2.set_ylabel("Vol (%)", color="white", fontsize=11)
        ax2.legend(fontsize=8, facecolor="#21262d", labelcolor="white", ncol=4)

        for ax in (ax1, ax2):
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_edgecolor("#30363d")

        plt.tight_layout()
        fig.savefig(reports_dir / "vol_regime_changes.png", dpi=150,
                    bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

        print(f"\n  Plots saved:")
        print(f"    reports/vol_forecast_vs_realized.png")
        print(f"    reports/vol_clusters.png")
        print(f"    reports/vol_cone.png")
        print(f"    reports/vol_regime_changes.png")

    except ImportError:
        print("\n  matplotlib not installed — skipping plots.")

    print(f"\n{sep}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
