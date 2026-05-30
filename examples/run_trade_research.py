"""
examples/run_trade_research.py
================================
End-to-End Trade Research & Capacity Analysis Demonstration.

Integrates Part 1 (Attribution), Part 2 (Transitions), and Part 3 (Capacity)
using synthetic data to produce institutional-grade research outputs.

Author: Saksham Mishra — AlgoQuant Club
"""
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from aqc.utils.logger import setup_logging
from aqc.diagnostics.trade_analyzer import TradeAnalyzer
from aqc.diagnostics.trade_attribution import TradeAttributionEngine
from aqc.diagnostics.trade_reports import TradeReportGenerator
from aqc.diagnostics.trade_visualization import TradeVisualizer

from aqc.research.regime_transitions import (
    TransitionEngine,
    TransitionAlphaAnalyzer,
    TransitionReportGenerator,
    TransitionVisualizer,
)

from aqc.execution.capacity_analyzer import CapacityAnalyzer, CapacityConfig
from aqc.execution.capacity_reports import CapacityReportGenerator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data Generation
# ---------------------------------------------------------------------------

def generate_synthetic_data(n_bars=252*5):
    """Generate 5 years of synthetic price and regime data."""
    idx = pd.date_range("2018-01-01", periods=n_bars, freq="D")
    
    # Prices (Geometric Brownian Motion with regime shifts)
    returns = np.random.normal(0.0005, 0.015, n_bars)
    prices = pd.Series(100 * np.exp(np.cumsum(returns)), index=idx)
    
    # Volatility
    vol = prices.pct_change().rolling(20).std() * np.sqrt(252)
    vol = vol.bfill()
    
    # Regimes
    vol_regime = pd.Series("NORMAL", index=idx)
    vol_regime[vol < 0.15] = "LOW"
    vol_regime[vol > 0.25] = "HIGH"
    vol_regime[vol > 0.40] = "EXTREME"
    
    trend_regime = pd.Series("RANGE_BOUND", index=idx)
    sma50 = prices.rolling(50).mean()
    sma200 = prices.rolling(200).mean()
    trend_regime[(prices > sma50) & (sma50 > sma200)] = "UPTREND"
    trend_regime[(prices < sma50) & (sma50 < sma200)] = "DOWNTREND"
    
    regime_df = pd.DataFrame({
        "vol_regime": vol_regime,
        "trend_regime": trend_regime
    }, index=idx)
    
    return prices, vol, regime_df

def generate_synthetic_trades(prices):
    """Generate a realistic trade log."""
    trades = []
    
    # Buy roughly every 10 days, hold for random duration
    for i in range(20, len(prices) - 20, 15):
        entry_idx = i
        hold_days = np.random.randint(1, 25)
        exit_idx = min(i + hold_days, len(prices) - 1)
        
        entry_ts = prices.index[entry_idx]
        exit_ts = prices.index[exit_idx]
        
        entry_px = prices.iloc[entry_idx]
        exit_px = prices.iloc[exit_idx]
        
        is_long = np.random.random() > 0.4
        side1, side2 = ("BUY", "SELL") if is_long else ("SELL", "BUY")
        qty = 1000
        
        strat = np.random.choice(["VWAP_Reversion", "Volume_Exhaustion", "ZScore", "Composite"])
        
        trades.append({
            "timestamp": entry_ts, "symbol": "XYZ", "side": side1,
            "quantity": qty, "fill_price": entry_px, "commission": 5.0,
            "realised_pnl": 0.0, "strategy_id": strat
        })
        
        # PnL
        pnl = (exit_px - entry_px) * qty - 10.0 if is_long else (entry_px - exit_px) * qty - 10.0
        
        trades.append({
            "timestamp": exit_ts, "symbol": "XYZ", "side": side2,
            "quantity": qty, "fill_price": exit_px, "commission": 5.0,
            "realised_pnl": pnl, "strategy_id": strat
        })
        
    return trades

# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def main():
    setup_logging()
    logger.info("Generating synthetic data...")
    prices, vol, regime_df = generate_synthetic_data()
    prices_df = pd.DataFrame({"XYZ": prices})
    trade_log = generate_synthetic_trades(prices)
    
    out_dir = Path("reports/research")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # -----------------------------------------------------------------------
    # PART 1: Trade Attribution
    # -----------------------------------------------------------------------
    print("\\n" + "="*80)
    print("  PART 1: TRADE ATTRIBUTION ENGINE")
    print("="*80)
    
    analyzer = TradeAnalyzer(trade_log, prices_df, regime_df, vol)
    trades_df = analyzer.to_dataframe()
    
    engine = TradeAttributionEngine(trades_df)
    attr_summary = engine.full_summary()
    
    rep_gen = TradeReportGenerator(trades_df, attr_summary, analyzer.trade_stats())
    rep_gen.print_report()
    rep_gen.save_all(out_dir)
    
    vis = TradeVisualizer(attr_summary, trades_df)
    vis.plot_all(out_dir / "plots")
    
    # -----------------------------------------------------------------------
    # PART 2: Regime Transition Alpha
    # -----------------------------------------------------------------------
    print("\\n" + "="*80)
    print("  PART 2: REGIME TRANSITION ALPHA")
    print("="*80)
    
    t_engine = TransitionEngine(regime_df, prices)
    events_df = t_engine.get_events_df()
    
    t_analyzer = TransitionAlphaAnalyzer(events_df, prices, horizons=[1, 3, 5, 10, 20])
    alpha_df = t_analyzer.analyze_alpha()
    
    t_rep = TransitionReportGenerator(alpha_df, horizons=[1, 3, 5, 10, 20])
    t_rep.print_report()
    t_rep.save_csv(out_dir)
    
    t_vis = TransitionVisualizer(t_analyzer)
    t_vis.plot_all(out_dir / "plots")
    
    # -----------------------------------------------------------------------
    # PART 3: Capacity Analysis
    # -----------------------------------------------------------------------
    print("\\n" + "="*80)
    print("  PART 3: CAPACITY ANALYSIS")
    print("="*80)
    
    # Synthetic backtest function for capacity analyzer
    def sim_capacity(cap, slippage, impact, liquidity):
        # Base stats
        sharpe = 2.5
        cagr = 0.20
        mdd = 0.10
        
        # As capital grows, impact scales by sqrt(cap) roughly
        # For a $1B fund, impact might destroy 80% of sharpe
        scale_factor = (cap / 1e6) ** 0.5  
        cost_penalty = scale_factor * 0.05
        
        new_sharpe = max(0.1, sharpe - cost_penalty)
        new_cagr = max(-0.05, cagr - (cost_penalty * 0.05))
        new_mdd = min(0.50, mdd + (cost_penalty * 0.02))
        
        total_cost = cap * (cost_penalty * 0.01)
        
        return {
            "sharpe": new_sharpe,
            "cagr": new_cagr,
            "mdd": new_mdd,
            "total_execution_cost": total_cost
        }
        
    config = CapacityConfig(capital_levels=[1e5, 1e6, 1e7, 5e7, 1e8, 5e8, 1e9])
    c_analyzer = CapacityAnalyzer(sim_capacity, config)
    cap_df = c_analyzer.run_capacity_analysis()
    max_cap = c_analyzer.detect_capacity_breakpoint(sharpe_decay_threshold=0.5)
    
    c_rep = CapacityReportGenerator(cap_df, max_cap)
    c_rep.print_report()
    c_rep.save_csv(out_dir)
    c_rep.plot_all(out_dir / "plots")
    
    print("\\n======================================================================")
    print(f"Research complete. All reports and plots saved to {out_dir}")
    print("======================================================================")

if __name__ == "__main__":
    main()
