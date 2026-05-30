"""
examples/run_regime_research.py
================================
End-to-End Regime Detection & Comparative Backtesting Research.

This script answers the core research question:

    "Do volatility targeting and regime awareness improve the risk-adjusted
     performance of AQC alpha strategies?"

Experiments:
    1. Baseline — Fixed sizing, no regime filter
    2. Vol-Targeted — Volatility-forecast-based position sizing
    3. Regime-Aware — Regime filter + vol-targeted sizing
    4. Combined — Multi-signal composite with all enhancements

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ── Setup ──────────────────────────────────────────────────────────────────
warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("regime_research")


# ===========================================================================
# 1. Synthetic Multi-Regime Data
# ===========================================================================


def generate_regime_data(n: int = 1000, seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    """Generate synthetic multi-regime OHLCV data.

    Creates data with distinct regimes:
    - Low-vol uptrend (bull market)
    - High-vol downtrend (bear market / crash)
    - Range-bound / mean-reverting
    - Recovery uptrend

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        OHLCV DataFrame and close price Series.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-02", periods=n, freq="B")

    # Regime 1: Low-vol uptrend (40%)
    n1 = n * 40 // 100
    r1 = rng.normal(0.0008, 0.006, n1)

    # Regime 2: High-vol crash (15%)
    n2 = n * 15 // 100
    r2 = rng.normal(-0.003, 0.025, n2)

    # Regime 3: Range-bound (25%)
    n3 = n * 25 // 100
    r3 = rng.normal(0.0, 0.008, n3)
    # Add mean-reversion component
    for i in range(1, len(r3)):
        r3[i] -= 0.15 * r3[i - 1]

    # Regime 4: Recovery (20%)
    n4 = n - n1 - n2 - n3
    r4 = rng.normal(0.001, 0.012, n4)

    returns = np.concatenate([r1, r2, r3, r4])
    prices = 100.0 * np.exp(np.cumsum(returns))

    spread = prices * 0.005
    ohlcv = pd.DataFrame({
        "open": prices * (1 + rng.normal(0, 0.001, n)),
        "high": prices + abs(rng.normal(0, 1, n)) * spread,
        "low": prices - abs(rng.normal(0, 1, n)) * spread,
        "close": prices,
        "volume": rng.uniform(1e6, 5e6, n),
    }, index=idx)

    return ohlcv, pd.Series(prices, index=idx, name="close")


# ===========================================================================
# 2. Backtest Simulation (Simplified Signal Engine)
# ===========================================================================


def simulate_strategy(
    prices: pd.Series,
    ohlc_df: pd.DataFrame,
    mode: str = "baseline",
    initial_capital: float = 100_000.0,
    seed: int = 42,
) -> tuple[pd.DataFrame, list[dict]]:
    """Simulate a mean-reversion strategy with different enhancements.

    Parameters
    ----------
    prices:
        Close price series.
    ohlc_df:
        OHLC DataFrame.
    mode:
        One of "baseline", "vol_target", "regime_aware", "combined".
    initial_capital:
        Starting cash.
    seed:
        Random seed.

    Returns
    -------
    tuple[pd.DataFrame, list[dict]]
        Equity curve and trade log.
    """
    from aqc.volatility.forecasting_engine import VolatilityForecastEngine
    from aqc.volatility.volatility_metrics import VolatilitySizer
    from aqc.regimes.regime_engine import RegimeEngine, RegimeFilter

    rng = np.random.default_rng(seed)
    n = len(prices)

    vol_engine = VolatilityForecastEngine()
    vol_sizer = VolatilitySizer(target_vol=0.10, risk_fraction=0.01)
    regime_engine = RegimeEngine(enable_hmm=False)
    regime_filter = RegimeFilter()

    # Mean-reversion signal: z-score of price vs 20-day MA
    ma20 = prices.rolling(20).mean()
    std20 = prices.rolling(20).std()
    zscore = (prices - ma20) / std20.replace(0, np.nan)
    zscore = zscore.fillna(0)

    equity = initial_capital
    position = 0.0
    avg_cost = 0.0
    cash = initial_capital

    equity_curve = []
    trade_log = []

    for i in range(50, n):
        price = float(prices.iloc[i])
        z = float(zscore.iloc[i])

        # Signal
        signal_strength = 0.0
        direction = "HOLD"

        if z < -1.5 and position <= 0:
            signal_strength = min(abs(z) / 3.0, 1.0)
            direction = "LONG"
        elif z > 1.5 and position >= 0:
            signal_strength = min(abs(z) / 3.0, 1.0)
            direction = "SHORT"
        elif position > 0 and z > 0:
            direction = "EXIT"
        elif position < 0 and z < 0:
            direction = "EXIT"

        # Position sizing
        if direction in ("LONG", "SHORT"):
            if mode == "baseline":
                qty = 100.0 * signal_strength
            else:
                # Vol-targeted sizing
                price_history = prices.iloc[max(0, i - 252) : i + 1]
                if len(price_history) > 50:
                    forecast = vol_engine.fit_and_forecast(price_history)
                    if forecast.forecast_vol > 0:
                        result = vol_sizer.size_position(
                            "SIM", price, forecast.forecast_vol, equity,
                        )
                        qty = max(1.0, float(result.quantity) * signal_strength)
                    else:
                        qty = 100.0 * signal_strength
                else:
                    qty = 100.0 * signal_strength

                # Regime filter
                if mode in ("regime_aware", "combined"):
                    price_history = prices.iloc[max(0, i - 252) : i + 1]
                    if len(price_history) > 60:
                        ohlc_slice = ohlc_df.iloc[max(0, i - 252) : i + 1]
                        snapshot = regime_engine.detect(price_history, ohlc_df=ohlc_slice)
                        if not regime_filter.should_trade("mean_reversion", snapshot):
                            direction = "HOLD"  # Block the signal

            if direction == "LONG" and position <= 0:
                # Close short if any
                if position < 0:
                    pnl = abs(position) * (avg_cost - price)
                    cash += pnl + abs(position) * avg_cost
                    trade_log.append({"realised_pnl": pnl})
                    position = 0
                    avg_cost = 0

                # Open long
                qty = min(qty, cash / price * 0.95)
                if qty >= 1:
                    position = qty
                    avg_cost = price
                    cash -= qty * price

            elif direction == "SHORT" and position >= 0:
                # Close long if any
                if position > 0:
                    pnl = position * (price - avg_cost)
                    cash += pnl + position * avg_cost
                    trade_log.append({"realised_pnl": pnl})
                    position = 0
                    avg_cost = 0

                # Open short (sell notional)
                qty = min(qty, cash / price * 0.5)
                if qty >= 1:
                    position = -qty
                    avg_cost = price
                    cash += qty * price  # receive cash from short sale

        elif direction == "EXIT" and position != 0:
            if position > 0:
                pnl = position * (price - avg_cost)
                cash += position * price
            else:
                pnl = abs(position) * (avg_cost - price)
                cash += pnl + abs(position) * avg_cost
            trade_log.append({"realised_pnl": pnl})
            position = 0
            avg_cost = 0

        # Mark to market
        equity = cash + position * price if position > 0 else cash
        if position < 0:
            equity = cash - abs(position) * price

        equity_curve.append({"equity": equity})

    eq_df = pd.DataFrame(equity_curve, index=prices.index[50:])
    return eq_df, trade_log


# ===========================================================================
# 3. Main Research Pipeline
# ===========================================================================


def main():
    """Run the full comparative analysis."""
    from aqc.regimes.regime_engine import RegimeEngine
    from aqc.research.comparison.comparator import BacktestComparator, StatisticalTests
    from aqc.research.comparison.reporting import ComparisonReportGenerator
    from aqc.portfolio.portfolio_metrics import PortfolioRiskMetrics

    print("=" * 70)
    print("AQC REGIME DETECTION & COMPARATIVE BACKTESTING RESEARCH")
    print("=" * 70)
    print()

    # ── Generate Data ──────────────────────────────────────────────────
    print("[1/5] Generating multi-regime synthetic data...")
    ohlc_df, prices = generate_regime_data(1000, seed=42)
    print(f"      {len(prices)} bars | {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"      Price range: ${prices.min():.2f} — ${prices.max():.2f}")
    print()

    # ── Regime Detection ───────────────────────────────────────────────
    print("[2/5] Running regime detection...")
    regime_engine = RegimeEngine(enable_hmm=True)
    regime_data = regime_engine.detect_full_series(prices, ohlc_df=ohlc_df)

    vol_counts = regime_data["vol_regime"].value_counts()
    trend_counts = regime_data["trend_regime"].value_counts()

    print("      Volatility Regimes:")
    for r, c in vol_counts.items():
        print(f"        {r:15s} : {c:4d} bars ({c/len(prices)*100:.1f}%)")
    print("      Trend Regimes:")
    for r, c in trend_counts.items():
        print(f"        {r:20s} : {c:4d} bars ({c/len(prices)*100:.1f}%)")
    print()

    # Save regime report
    Path("reports").mkdir(exist_ok=True)
    regime_data.to_csv("reports/regime_detection_report.csv")
    print("      Saved: reports/regime_detection_report.csv")
    print()

    # ── Run Experiments ────────────────────────────────────────────────
    print("[3/5] Running comparative backtests...")
    comparator = BacktestComparator()

    experiments = [
        ("Baseline (Fixed)", "baseline"),
        ("Vol-Targeted", "vol_target"),
        ("Regime-Aware", "regime_aware"),
        ("Combined", "combined"),
    ]

    for name, mode in experiments:
        print(f"      Running: {name}...")
        eq, trades = simulate_strategy(prices, ohlc_df, mode=mode)
        comparator.add_result(name, eq, trades)
    print()

    # ── Compare ────────────────────────────────────────────────────────
    print("[4/5] Comparative Analysis")
    print("-" * 70)

    comparison = comparator.compare()
    key_metrics = [
        "sharpe_ratio", "sortino_ratio", "cagr", "max_drawdown_pct",
        "calmar_ratio", "annualised_volatility", "win_rate",
        "profit_factor", "total_return_pct",
    ]

    available = [m for m in key_metrics if m in comparison.index]
    print(comparison.loc[available].to_string())
    print()

    # Portfolio risk metrics
    print("Portfolio Risk Metrics:")
    for name in comparator.results:
        returns = comparator.get_returns(name)
        if len(returns) > 10:
            prm = PortfolioRiskMetrics(returns)
            var = prm.historical_var()
            es = prm.expected_shortfall()
            vol = prm.portfolio_volatility()
            print(f"  {name:25s} | Vol={vol:.4f} | VaR={var:.4f} | ES={es:.4f}")
    print()

    # ── Statistical Tests ──────────────────────────────────────────────
    print("Statistical Significance Tests:")
    print("-" * 70)

    baseline_returns = comparator.get_returns("Baseline (Fixed)")
    tests = StatisticalTests()

    for name in ["Vol-Targeted", "Regime-Aware", "Combined"]:
        enhanced_returns = comparator.get_returns(name)
        if len(enhanced_returns) > 10 and len(baseline_returns) > 10:
            # T-test
            t_result = tests.t_test_returns(enhanced_returns, baseline_returns)
            # Sharpe diff
            sharpe_result = tests.sharpe_difference_test(
                enhanced_returns, baseline_returns,
            )
            # Bootstrap CI
            ci = tests.bootstrap_sharpe_ci(enhanced_returns)

            print(f"  {name} vs Baseline:")
            print(f"    t-stat={t_result['t_stat']:.4f}, p={t_result['p_value']:.4f}, sig={t_result['significant_5pct']}")
            print(f"    Sharpe diff={sharpe_result['sharpe_diff']:.4f}, p={sharpe_result['p_value']:.4f}, sig={sharpe_result['significant_5pct']}")
            print(f"    Sharpe 95% CI: [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")
            print()

    # ── Visualisations ─────────────────────────────────────────────────
    print("[5/5] Generating plots and reports...")
    reporter = ComparisonReportGenerator(
        comparator, regime_data=regime_data, output_dir="reports",
    )
    reporter.generate_all()
    reporter.save_regime_report()

    # Save vol targeting report
    comparison.to_csv("reports/volatility_targeting_report.csv")
    print("      Saved: reports/volatility_targeting_report.csv")
    print()

    # ── Research Conclusion ────────────────────────────────────────────
    print("=" * 70)
    print("RESEARCH CONCLUSION")
    print("=" * 70)
    print()

    baseline_sharpe = comparison.loc["sharpe_ratio", "Baseline (Fixed)"]
    best_name = None
    best_sharpe = baseline_sharpe

    for name in ["Vol-Targeted", "Regime-Aware", "Combined"]:
        if name in comparison.columns:
            s = comparison.loc["sharpe_ratio", name]
            if not np.isnan(s) and s > best_sharpe:
                best_sharpe = s
                best_name = name

    if best_name:
        improvement = (best_sharpe - baseline_sharpe)
        print(f"  Best performer: {best_name}")
        print(f"  Sharpe improvement: {baseline_sharpe:.4f} -> {best_sharpe:.4f} (Delta={improvement:+.4f})")

        sharpe_result = tests.sharpe_difference_test(
            comparator.get_returns(best_name), baseline_returns,
        )
        if sharpe_result["significant_5pct"]:
            print(f"  Statistical significance: YES (p={sharpe_result['p_value']:.4f})")
            print()
            print("  VERDICT: Volatility targeting and/or regime awareness")
            print("  provide STATISTICALLY SIGNIFICANT improvements in")
            print("  risk-adjusted performance.")
        else:
            print(f"  Statistical significance: NO (p={sharpe_result['p_value']:.4f})")
            print()
            print("  VERDICT: Improvements are directionally positive but not")
            print("  statistically significant at the 5% level on this sample.")
            print("  Larger sample size or out-of-sample validation recommended.")
    else:
        print("  Baseline performed best on this sample.")
        print("  Enhanced methods did not improve risk-adjusted returns.")

    print()
    print("=" * 70)
    print("Research complete. All reports saved to reports/")
    print("=" * 70)


if __name__ == "__main__":
    main()
