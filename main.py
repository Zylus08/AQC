"""
main.py
=======
AQC — AlgoQuant Club Backtest Framework — Entry Point.

Usage
-----
Run with the default configuration::

    python main.py

Run with a custom configuration file::

    python main.py --config configs/my_config.yaml

Run with additional overrides::

    python main.py --strategy rsi_mean_reversion --capital 500000

When no CSV data file is found, the engine automatically generates synthetic
OHLCV data so the framework can be tested without a real data source.

Author: AQC Team
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project imports (after path is confirmed)
# ---------------------------------------------------------------------------
from aqc.utils.logger import setup_logging
from aqc.utils.config_loader import load_config
from aqc.backtester.event_queue import EventQueue
from aqc.backtester.broker import (
    SimulatedBroker,
    PercentageCommission,
    FlatFeeCommission,
    ZeroCommission,
    FixedBpsSlippage,
    ZeroSlippage,
)
from aqc.backtester.execution import ExecutionEngine
from aqc.backtester.portfolio import Portfolio
from aqc.backtester.engine import BacktestEngine
from aqc.risk.risk_manager import RiskManager, RiskConfig
from aqc.strategies.sample_strategy import (
    SMACrossoverStrategy,
    RSIMeanReversionStrategy,
    EMAMomentumStrategy,
)
from aqc.data.loaders.csv_loader import CSVDataLoader, DataLoaderError
from aqc.analytics.reporting import ReportGenerator

logger = logging.getLogger("aqc.main")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="AQC — AlgoQuant Club Backtest Framework",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file (default: configs/config.yaml)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="Override strategy name: sma_crossover | rsi_mean_reversion | ema_momentum",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=None,
        help="Override initial capital",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Override symbol (single symbol override)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Strategy factory
# ---------------------------------------------------------------------------

def build_strategy(
    name: str,
    event_queue: EventQueue,
    symbols: list[str],
    params: dict,
):
    """Instantiate the configured strategy.

    Parameters
    ----------
    name:
        Strategy identifier string.
    event_queue:
        Shared event queue.
    symbols:
        Instrument list.
    params:
        Strategy parameter dictionary from config.

    Returns
    -------
    BaseStrategy
    """
    strategies = {
        "sma_crossover": lambda: SMACrossoverStrategy(
            event_queue=event_queue,
            symbols=symbols,
            fast_period=int(params.get("fast_period", 20)),
            slow_period=int(params.get("slow_period", 50)),
        ),
        "rsi_mean_reversion": lambda: RSIMeanReversionStrategy(
            event_queue=event_queue,
            symbols=symbols,
            rsi_period=int(params.get("rsi_period", 14)),
            oversold=float(params.get("oversold", 30)),
            overbought=float(params.get("overbought", 70)),
            allow_short=bool(params.get("allow_short", False)),
        ),
        "ema_momentum": lambda: EMAMomentumStrategy(
            event_queue=event_queue,
            symbols=symbols,
            short_period=int(params.get("short_period", 9)),
            medium_period=int(params.get("medium_period", 21)),
            long_period=int(params.get("long_period", 50)),
        ),
    }

    if name not in strategies:
        raise ValueError(
            f"Unknown strategy '{name}'. Available: {list(strategies.keys())}"
        )
    return strategies[name]()


# ---------------------------------------------------------------------------
# Commission / slippage factories
# ---------------------------------------------------------------------------

def build_commission(broker_cfg: dict):
    """Build the commission model from config."""
    model = broker_cfg.get("commission_model", "percentage")
    if model == "percentage":
        return PercentageCommission(rate=float(broker_cfg.get("commission_rate", 0.001)))
    elif model == "flat_fee":
        return FlatFeeCommission(fee=float(broker_cfg.get("commission_flat_fee", 5.0)))
    elif model == "zero":
        return ZeroCommission()
    else:
        logger.warning("Unknown commission model '%s' — using zero.", model)
        return ZeroCommission()


def build_slippage(broker_cfg: dict):
    """Build the slippage model from config."""
    model = broker_cfg.get("slippage_model", "fixed_bps")
    if model == "fixed_bps":
        return FixedBpsSlippage(bps=float(broker_cfg.get("slippage_bps", 5)))
    elif model == "zero":
        return ZeroSlippage()
    else:
        logger.warning("Unknown slippage model '%s' — using zero.", model)
        return ZeroSlippage()


# ---------------------------------------------------------------------------
# Synthetic data generator (for demo / CI)
# ---------------------------------------------------------------------------

def generate_synthetic_data(
    symbol: str,
    n_bars: int = 500,
    start: str = "2022-01-03",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic OHLCV data using geometric Brownian motion.

    Parameters
    ----------
    symbol:
        Instrument label (used in log messages only).
    n_bars:
        Number of daily bars to generate.
    start:
        Start date string (ISO format).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Synthetic OHLCV DataFrame with a DatetimeIndex.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_bars)

    # GBM parameters
    mu = 0.0003          # daily drift ≈ 7.5% annual
    sigma = 0.012        # daily vol ≈ 19% annual
    s0 = 150.0           # starting price

    # Simulate close prices
    returns = rng.normal(mu, sigma, n_bars)
    log_prices = np.log(s0) + np.cumsum(returns)
    closes = np.exp(log_prices)

    # Derive OHLV from closes
    noise = rng.uniform(0.001, 0.008, n_bars)
    highs = closes * (1 + noise)
    lows = closes * (1 - noise)
    opens = np.roll(closes, 1)
    opens[0] = s0
    volumes = rng.integers(1_000_000, 10_000_000, n_bars).astype(float)

    df = pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        },
        index=pd.DatetimeIndex(dates, name="timestamp"),
    )

    logger.info(
        "Generated %d synthetic bars for %s [%s to %s]",
        n_bars,
        symbol,
        dates[0].date(),
        dates[-1].date(),
    )
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """Framework entry point.

    Returns
    -------
    int
        Exit code (0 = success, 1 = error).
    """
    args = parse_args()

    # 1. Load configuration
    config = load_config(args.config)

    # Apply CLI overrides
    if args.capital:
        config["backtest"]["initial_capital"] = args.capital
    if args.strategy:
        config["strategy"]["name"] = args.strategy
    if args.symbol:
        config["backtest"]["symbols"] = [args.symbol]

    # 2. Setup logging (must be before any module-level logger calls)
    log_cfg = config["logging"]
    setup_logging(
        level=log_cfg.get("level", "INFO"),
        log_dir=log_cfg.get("log_dir", "logs"),
        log_filename=log_cfg.get("log_filename", "backtest.log"),
        log_to_file=log_cfg.get("log_to_file", True),
    )

    logger.info("AQC AlgoQuant Club — Backtest Engine starting")
    logger.info("Strategy : %s", config["strategy"]["name"])
    logger.info("Capital  : %.2f", config["backtest"]["initial_capital"])
    logger.info("Symbols  : %s", config["backtest"]["symbols"])

    # 3. Load market data
    symbols: list[str] = config["backtest"]["symbols"]
    data_cfg = config["data"]
    data_dir = data_cfg.get("data_dir", "data/raw")
    data: dict[str, pd.DataFrame] = {}

    loader = CSVDataLoader(
        data_dir=data_dir,
        fill_method=data_cfg.get("fill_method", "ffill"),
        tz=data_cfg.get("timezone"),
    )

    for symbol in symbols:
        csv_path = Path(data_dir) / f"{symbol}.csv"
        try:
            df = loader.load(str(csv_path), symbol=symbol)
            # Apply date filter if specified
            start = config["backtest"].get("start_date")
            end = config["backtest"].get("end_date")
            if start:
                df = df[df.index >= pd.Timestamp(start)]
            if end:
                df = df[df.index <= pd.Timestamp(end)]
            if len(df) == 0:
                raise DataLoaderError("No bars remaining after date filter.")
            data[symbol] = df
        except DataLoaderError as exc:
            logger.warning(
                "Could not load CSV for %s (%s) — generating synthetic data.", symbol, exc
            )
            data[symbol] = generate_synthetic_data(symbol, n_bars=500)

    if not data:
        logger.error("No data available — aborting.")
        return 1

    # 4. Build framework components
    event_queue = EventQueue()

    risk_cfg = config["risk"]
    risk_config = RiskConfig(
        max_position_size=risk_cfg.get("max_position_size", float("inf")),
        max_position_pct_equity=float(risk_cfg.get("max_position_pct_equity", 0.20)),
        max_gross_exposure_pct=float(risk_cfg.get("max_gross_exposure_pct", 1.0)),
        max_daily_loss_pct=float(risk_cfg.get("max_daily_loss_pct", 0.05)),
        max_open_positions=int(risk_cfg.get("max_open_positions", 10)),
    )
    risk_manager = RiskManager(config=risk_config)

    initial_capital = float(config["backtest"]["initial_capital"])
    portfolio = Portfolio(
        event_queue=event_queue,
        risk_manager=risk_manager,
        initial_capital=initial_capital,
        default_quantity=float(config["portfolio"].get("default_quantity", 100)),
    )

    # Seed daily risk state
    risk_manager.reset_daily_state(initial_capital)

    broker_cfg = config["broker"]
    commission = build_commission(broker_cfg)
    slippage = build_slippage(broker_cfg)
    broker = SimulatedBroker(
        event_queue=event_queue,
        commission_model=commission,
        slippage_model=slippage,
        exchange=broker_cfg.get("exchange", "SIMULATED"),
    )
    execution_engine = ExecutionEngine(broker=broker, event_queue=event_queue)

    strategy_cfg = config["strategy"]
    strategy = build_strategy(
        name=strategy_cfg["name"],
        event_queue=event_queue,
        symbols=symbols,
        params=strategy_cfg.get("params", {}),
    )

    logger.info("Strategy  : %s", strategy)
    logger.info("Broker    : %s", broker)
    logger.info("Risk      : %s", risk_manager)

    # 5. Run backtest
    engine = BacktestEngine(
        data=data,
        strategy=strategy,
        portfolio=portfolio,
        execution_engine=execution_engine,
        event_queue=event_queue,
        config=config,
    )
    results = engine.run()

    # 6. Persist outputs
    out_cfg = config.get("output", {})
    reports_dir = out_cfg.get("reports_dir", "reports")

    reporter = ReportGenerator(
        portfolio_summary=results["portfolio_summary"],
        performance_metrics=results["performance_metrics"],
        equity_curve=results["equity_curve"],
        trade_log=results["trade_log"],
    )

    if out_cfg.get("save_report", True):
        reporter.save_report(f"{reports_dir}/backtest_report.txt")

    if out_cfg.get("export_equity_curve", True):
        reporter.export_equity_curve(f"{reports_dir}/equity_curve.csv")

    if out_cfg.get("export_trade_log", True):
        reporter.export_trade_log(f"{reports_dir}/trade_log.csv")

    logger.info(
        "Run complete — %d bars in %.3f s",
        results["bars_processed"],
        results["elapsed_seconds"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
