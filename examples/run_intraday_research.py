"""
examples/run_intraday_research.py
==================================
Intraday Mean Reversion Research Report Generator.

Runs all four intraday strategies on synthetic data and produces:
- Signal statistics
- Trade statistics
- Performance comparison table
- Alpha decay analysis
- Return distribution plots
- Per-strategy equity curves

Usage::

    python examples/run_intraday_research.py

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
from aqc.utils.logger import setup_logging

setup_logging(level="WARNING", log_dir="logs", log_to_file=True)
logger = logging.getLogger("aqc.examples.intraday")
logger.setLevel(logging.INFO)

from aqc.backtester.event_queue import EventQueue
from aqc.backtester.broker import (
    SimulatedBroker, PercentageCommission, FixedBpsSlippage,
)
from aqc.backtester.execution import ExecutionEngine
from aqc.backtester.portfolio import Portfolio
from aqc.backtester.engine import BacktestEngine
from aqc.risk.risk_manager import RiskManager, RiskConfig

from aqc.strategies.intraday import (
    VWAPReversionStrategy,
    VolumeExhaustionStrategy,
    ZScoreReversionStrategy,
    CompositeMeanReversionStrategy,
)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def generate_mean_reverting_data(
    symbol: str = "AAPL",
    n_bars: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic data with mean-reverting characteristics.

    Uses an Ornstein-Uhlenbeck process to create data where prices
    naturally revert to a moving mean — ideal for testing MR strategies.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start="2022-01-03", periods=n_bars)

    # OU process parameters
    theta = 0.05   # mean-reversion speed
    mu = 5.0       # log-price mean
    sigma = 0.015  # volatility

    log_prices = np.zeros(n_bars)
    log_prices[0] = mu

    for i in range(1, n_bars):
        dW = rng.normal(0, 1)
        log_prices[i] = (
            log_prices[i - 1]
            + theta * (mu - log_prices[i - 1])
            + sigma * dW
        )

    closes = np.exp(log_prices)
    noise = rng.uniform(0.002, 0.008, n_bars)
    highs = closes * (1 + noise)
    lows = closes * (1 - noise)
    opens = np.roll(closes, 1)
    opens[0] = closes[0]

    # Add volume spikes at random intervals
    base_vol = rng.integers(1_000_000, 5_000_000, n_bars).astype(float)
    spike_indices = rng.choice(n_bars, size=n_bars // 15, replace=False)
    base_vol[spike_indices] *= rng.uniform(2.5, 5.0, len(spike_indices))

    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": base_vol},
        index=pd.DatetimeIndex(dates, name="timestamp"),
    )


# ---------------------------------------------------------------------------
# Backtest runner
# ---------------------------------------------------------------------------


def run_backtest(
    strategy_cls,
    data: dict[str, pd.DataFrame],
    params: dict,
    capital: float = 100_000,
    qty: float = 100,
) -> dict:
    """Run a full backtest and return results."""
    import aqc.analytics.reporting as rep_module

    eq = EventQueue()
    risk = RiskManager(
        config=RiskConfig(
            max_position_pct_equity=1.0,
            max_gross_exposure_pct=5.0,
            max_daily_loss_pct=0.99,
            max_open_positions=20,
        )
    )
    portfolio = Portfolio(
        event_queue=eq, risk_manager=risk,
        initial_capital=capital, default_quantity=qty,
    )
    risk.reset_daily_state(capital)
    broker = SimulatedBroker(
        event_queue=eq,
        commission_model=PercentageCommission(rate=0.001),
        slippage_model=FixedBpsSlippage(bps=5),
    )
    exec_engine = ExecutionEngine(broker=broker, event_queue=eq)
    strategy = strategy_cls(
        event_queue=eq,
        symbols=list(data.keys()),
        **params,
    )

    # Suppress report printout
    original_print = rep_module.ReportGenerator.print_report
    rep_module.ReportGenerator.print_report = lambda self: None

    engine_logger = logging.getLogger("aqc.backtester.engine")
    old_level = engine_logger.level
    engine_logger.setLevel(logging.WARNING)

    engine = BacktestEngine(
        data=data, strategy=strategy, portfolio=portfolio,
        execution_engine=exec_engine, event_queue=eq,
    )
    result = engine.run()

    rep_module.ReportGenerator.print_report = original_print
    engine_logger.setLevel(old_level)

    return result


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------


def compute_trade_stats(trade_log: list[dict]) -> dict:
    """Compute detailed trade statistics."""
    if not trade_log:
        return {"n_trades": 0}
    df = pd.DataFrame(trade_log)
    pnl = df.get("realised_pnl", pd.Series(dtype=float)).fillna(0)
    closed = pnl[pnl != 0]
    if closed.empty:
        return {"n_trades": 0}

    wins = closed[closed > 0]
    losses = closed[closed < 0]

    return {
        "n_trades": len(closed),
        "n_wins": len(wins),
        "n_losses": len(losses),
        "win_rate": round(len(wins) / len(closed) * 100, 1) if len(closed) > 0 else 0,
        "avg_win": round(float(wins.mean()), 2) if len(wins) > 0 else 0,
        "avg_loss": round(float(losses.mean()), 2) if len(losses) > 0 else 0,
        "total_pnl": round(float(closed.sum()), 2),
        "max_win": round(float(wins.max()), 2) if len(wins) > 0 else 0,
        "max_loss": round(float(losses.min()), 2) if len(losses) > 0 else 0,
        "profit_factor": round(float(wins.sum() / abs(losses.sum())), 2) if len(losses) > 0 and losses.sum() != 0 else float("inf"),
        "avg_holding": "N/A",  # would need bar timestamps
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    import math

    sep = "=" * 70
    thin = "-" * 70

    print(f"\n{sep}")
    print("  AQC INTRADAY MEAN REVERSION RESEARCH REPORT")
    print(sep)

    # Generate data
    data = {"AAPL": generate_mean_reverting_data("AAPL", n_bars=500, seed=42)}
    n_bars = len(data["AAPL"])
    print(f"\n  Data: {n_bars} bars of OU-process synthetic data")
    print(f"  Date range: {data['AAPL'].index[0].date()} to {data['AAPL'].index[-1].date()}")

    # Define strategies and parameters
    strategies = {
        "VWAP Reversion": (
            VWAPReversionStrategy,
            {"entry_threshold": 1.5, "exit_threshold": 0.3, "rolling_window": 15,
             "max_holding_bars": 15, "stop_loss_pct": 0.02},
        ),
        "Volume Exhaustion": (
            VolumeExhaustionStrategy,
            {"spike_mult": 2.0, "breakout_window": 8, "volume_window": 15,
             "require_wick_rejection": False, "max_holding_bars": 12, "stop_loss_pct": 0.015},
        ),
        "Z-Score Adaptive": (
            ZScoreReversionStrategy,
            {"z_window": 15, "base_entry_z": 1.5, "base_exit_z": 0.3,
             "vol_lookback": 40, "vol_adjustment": 0.5,
             "max_holding_bars": 15, "stop_loss_pct": 0.02, "allow_short": True},
        ),
        "Composite Alpha": (
            CompositeMeanReversionStrategy,
            {"w_vwap": 0.4, "w_volume": 0.3, "w_zscore": 0.3,
             "composite_threshold": 0.2, "exit_threshold": 0.05,
             "min_signals": 1, "max_holding_bars": 20, "stop_loss_pct": 0.02},
        ),
    }

    results_summary = []

    for name, (cls, params) in strategies.items():
        logger.info("Running %s...", name)
        result = run_backtest(cls, data, params)
        perf = result.get("performance_metrics", {})
        trade_stats = compute_trade_stats(result.get("trade_log", []))

        results_summary.append({
            "Strategy": name,
            "Sharpe": perf.get("sharpe_ratio", float("nan")),
            "Sortino": perf.get("sortino_ratio", float("nan")),
            "Return%": perf.get("total_return_pct", float("nan")),
            "MaxDD%": perf.get("max_drawdown_pct", float("nan")),
            "Trades": trade_stats.get("n_trades", 0),
            "WinRate%": trade_stats.get("win_rate", 0),
            "PF": trade_stats.get("profit_factor", float("nan")),
            "TotalPnL": trade_stats.get("total_pnl", 0),
        })

    # --- Performance Comparison Table ---
    print(f"\n{thin}")
    print("  STRATEGY PERFORMANCE COMPARISON")
    print(thin)

    df = pd.DataFrame(results_summary)
    # Format for display
    print()
    header = f"  {'Strategy':<22} {'Sharpe':>8} {'Sortino':>8} {'Ret%':>8} {'MaxDD%':>8} {'Trades':>7} {'Win%':>6} {'PF':>6} {'PnL':>10}"
    print(header)
    print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*7} {'-'*6} {'-'*6} {'-'*10}")

    for _, row in df.iterrows():
        def fmt(v, w=8, d=4):
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return f"{'N/A':>{w}}"
            return f"{v:>{w}.{d}f}" if isinstance(v, float) else f"{v:>{w}}"

        print(
            f"  {row['Strategy']:<22} "
            f"{fmt(row['Sharpe'])} "
            f"{fmt(row['Sortino'])} "
            f"{fmt(row['Return%'], d=2)} "
            f"{fmt(row['MaxDD%'], d=2)} "
            f"{fmt(row['Trades'], 7, 0)} "
            f"{fmt(row['WinRate%'], 6, 1)} "
            f"{fmt(row['PF'], 6, 2)} "
            f"{fmt(row['TotalPnL'], 10, 2)}"
        )

    # --- Save CSV report ---
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    df.to_csv(reports_dir / "intraday_comparison.csv", index=False)
    print(f"\n  Report saved: reports/intraday_comparison.csv")

    # --- Generate plots ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.style.use("dark_background")

        # Re-run strategies to collect equity curves
        equity_curves = {}
        for name, (cls, params) in strategies.items():
            result = run_backtest(cls, data, params)
            eq_curve = result.get("equity_curve", pd.DataFrame())
            if not eq_curve.empty:
                equity_curves[name] = eq_curve

        # --- Plot 1: Equity curves ---
        PALETTE = ["#4FC3F7", "#81C784", "#FFB74D", "#F06292"]
        fig, ax = plt.subplots(figsize=(14, 7), facecolor="#0d1117")
        ax.set_facecolor("#161b22")

        for i, (name, curve) in enumerate(equity_curves.items()):
            if "equity" in curve.columns:
                normalised = curve["equity"] / curve["equity"].iloc[0] * 100
                ax.plot(normalised.index, normalised.values,
                        color=PALETTE[i % len(PALETTE)],
                        linewidth=1.5, label=name)

        ax.axhline(100, color="#555", linewidth=1, linestyle="--", alpha=0.7)
        ax.set_title("Intraday Mean Reversion — Equity Curves",
                      color="white", fontsize=14, fontweight="bold")
        ax.set_ylabel("Equity (rebased to 100)", color="white", fontsize=11)
        ax.set_xlabel("Date", color="white", fontsize=11)
        ax.tick_params(colors="white")
        ax.legend(fontsize=10, facecolor="#21262d", labelcolor="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        plt.tight_layout()
        fig.savefig(reports_dir / "intraday_equity_curves.png",
                    dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

        # --- Plot 2: Return distributions ---
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor="#0d1117")
        for i, (name, (cls, params)) in enumerate(strategies.items()):
            ax = axes[i // 2][i % 2]
            ax.set_facecolor("#161b22")
            result = run_backtest(cls, data, params)
            trades = pd.DataFrame(result.get("trade_log", []))
            pnl = trades.get("realised_pnl", pd.Series(dtype=float)).fillna(0)
            closed = pnl[pnl != 0]

            if not closed.empty:
                ax.hist(closed, bins=max(5, len(closed) // 3),
                        color=PALETTE[i], alpha=0.8, edgecolor="#21262d")
                ax.axvline(float(closed.mean()), color="white",
                           linewidth=1.5, linestyle="--",
                           label=f"Mean: {closed.mean():.1f}")
                ax.axvline(0, color="#888", linewidth=1, linestyle=":")
                ax.legend(fontsize=8, facecolor="#21262d", labelcolor="white")
            else:
                ax.text(0.5, 0.5, "No closed trades", transform=ax.transAxes,
                        ha="center", color="white", fontsize=11)

            ax.set_title(name, color="white", fontsize=11)
            ax.tick_params(colors="white", labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor("#30363d")

        fig.suptitle("Trade Return Distributions",
                     color="white", fontsize=14, fontweight="bold")
        plt.tight_layout()
        fig.savefig(reports_dir / "intraday_return_distributions.png",
                    dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

        # --- Plot 3: Signal statistics heatmap ---
        fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0d1117")
        ax.set_facecolor("#161b22")

        metrics_for_heatmap = ["Sharpe", "Sortino", "Return%", "MaxDD%", "WinRate%", "PF"]
        heat_data = df.set_index("Strategy")[metrics_for_heatmap].astype(float)

        # Replace inf/nan for display
        heat_data = heat_data.replace([np.inf, -np.inf], np.nan).fillna(0)

        im = ax.imshow(heat_data.values, aspect="auto", cmap="RdYlGn")
        ax.set_yticks(range(len(heat_data)))
        ax.set_yticklabels(heat_data.index, color="white", fontsize=10)
        ax.set_xticks(range(len(metrics_for_heatmap)))
        ax.set_xticklabels(metrics_for_heatmap, color="white", fontsize=10)

        for row_i in range(len(heat_data)):
            for col_j in range(len(metrics_for_heatmap)):
                val = heat_data.values[row_i, col_j]
                ax.text(col_j, row_i, f"{val:.2f}",
                        ha="center", va="center", fontsize=9, color="black",
                        fontweight="bold")

        ax.set_title("Strategy Metrics Heatmap",
                      color="white", fontsize=13, fontweight="bold")
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.ax.tick_params(labelcolor="white", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        plt.tight_layout()
        fig.savefig(reports_dir / "intraday_metrics_heatmap.png",
                    dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

        print("  Plots saved:")
        print("    reports/intraday_equity_curves.png")
        print("    reports/intraday_return_distributions.png")
        print("    reports/intraday_metrics_heatmap.png")

    except ImportError:
        print("  matplotlib not installed — skipping plots.")

    print(f"\n{sep}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
