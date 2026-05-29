"""
examples/run_walk_forward.py
============================
Example script demonstrating the Walk-Forward Optimisation Framework.

Runs a full WFO study on synthetic data with the SMA Crossover strategy,
exports CSV results, generates plots, and prints a validation report.

Usage::

    python examples/run_walk_forward.py

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Setup logging
# ---------------------------------------------------------------------------
from aqc.utils.logger import setup_logging

setup_logging(level="INFO", log_dir="logs", log_to_file=True)
logger = logging.getLogger("aqc.examples.wfo")

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from aqc.research import (
    WalkForwardEngine,
    WalkForwardMode,
    ParameterSpace,
    IntParam,
    FloatParam,
    ObjectiveMetric,
    WalkForwardValidator,
)
from aqc.strategies.sample_strategy import SMACrossoverStrategy, RSIMeanReversionStrategy


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------


def generate_synthetic_data(
    symbol: str = "AAPL",
    n_bars: int = 600,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic OHLCV data using geometric Brownian motion."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start="2020-01-02", periods=n_bars)

    mu = 0.0004
    sigma = 0.015
    s0 = 150.0

    returns = rng.normal(mu, sigma, n_bars)
    log_prices = np.log(s0) + np.cumsum(returns)
    closes = np.exp(log_prices)

    noise = rng.uniform(0.001, 0.008, n_bars)
    highs = closes * (1 + noise)
    lows = closes * (1 - noise)
    opens = np.roll(closes, 1)
    opens[0] = s0
    volumes = rng.integers(1_000_000, 10_000_000, n_bars).astype(float)

    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=pd.DatetimeIndex(dates, name="timestamp"),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    logger.info("=" * 65)
    logger.info("AQC Walk-Forward Optimisation Demo")
    logger.info("=" * 65)

    # 1. Generate synthetic data
    data = {"AAPL": generate_synthetic_data("AAPL", n_bars=600, seed=42)}
    logger.info("Generated %d bars of synthetic data", len(data["AAPL"]))

    # 2. Define parameter search space
    space = ParameterSpace()
    space.add(IntParam("fast_period", low=5, high=25, step=5))    # [5, 10, 15, 20, 25]
    space.add(IntParam("slow_period", low=30, high=60, step=10))  # [30, 40, 50, 60]

    logger.info("Parameter space: %d combinations", space.grid_size())

    # 3. Configure and run the Walk-Forward Engine
    engine = WalkForwardEngine(
        data=data,
        strategy_factory=SMACrossoverStrategy,
        parameter_space=space,
        mode=WalkForwardMode.ROLLING,
        train_period=150,           # 150 bars (~6 months)
        test_period=75,             # 75 bars (~3 months)
        n_folds=4,
        optimizer="grid",
        objective=ObjectiveMetric.SHARPE,
        initial_capital=100_000,
        default_quantity=100,
    )

    logger.info("Engine configured: %s", engine)

    # Run the WFO
    result = engine.run()

    # 4. Export CSV results
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    result.save_csv("reports/walk_forward_results.csv")

    # 5. Validate and generate plots
    validator = WalkForwardValidator(result=result, output_dir="reports")

    # Print report
    validator.print_report()
    validator.save_report("walk_forward_report.txt")

    # Generate plots
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend for CI/scripts

        validator.plot_equity_curves(save=True)
        validator.plot_sharpe_distribution(save=True)
        validator.plot_parameter_stability(save=True)
        logger.info("All plots saved to reports/")
    except ImportError:
        logger.warning("matplotlib not installed — skipping plots.")

    # 6. Print summary
    agg = result.aggregate_metrics()
    stab = result.parameter_stability()

    print("\n" + "=" * 65)
    print("  DEMO SUMMARY")
    print("=" * 65)
    print(f"  Folds completed     : {agg['n_folds']}")
    print(f"  Mean OOS Sharpe     : {agg.get('test_sharpe_ratio_mean', 'N/A'):.4f}")
    print(f"  Mean OOS Return     : {agg.get('test_total_return_pct_mean', 'N/A'):.2f}%")
    print(f"  IS/OOS Correlation  : {validator.is_oos_correlation():.4f}")
    print(f"  Overfitting Score   : {validator.overfitting_score():.4f}")
    print()
    print("  Parameter Stability:")
    for name, stats in stab.items():
        if "cv" in stats:
            print(f"    {name:<16}: mean={stats['mean']:.1f}  std={stats['std']:.1f}  cv={stats['cv']:.3f}")
    print()
    print("  Output files:")
    for f in sorted(reports_dir.glob("*walk_forward*")):
        print(f"    {f}")
    for f in sorted(reports_dir.glob("*wfo*")):
        print(f"    {f}")
    print("=" * 65)

    return 0


if __name__ == "__main__":
    sys.exit(main())
