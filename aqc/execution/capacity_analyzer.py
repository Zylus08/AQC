"""
aqc/execution/capacity_analyzer.py
====================================
Determine whether alpha survives larger capital allocations by simulating
execution constraints, slippage, and market impact across AUM scenarios.

Author: Saksham Mishra — AlgoQuant Club
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd

from aqc.execution.liquidity_model import LiquidityModel
from aqc.execution.market_impact import SquareRootImpactModel
from aqc.execution.slippage_model import SlippageModel

logger = logging.getLogger(__name__)


@dataclass
class CapacityConfig:
    """Configuration for capacity analysis scenarios."""
    capital_levels: list[float]
    fixed_slippage_bps: float = 0.0
    impact_coefficient: float = 0.1
    max_participation: float = 0.10


class CapacityAnalyzer:
    """Analyze strategy capacity constraints across different capital bases.

    Parameters
    ----------
    simulation_fn : Callable
        A function `sim_fn(capital, slippage_model, impact_model, liquidity_model)`
        that runs the backtest and returns a dict with metrics:
        {"equity": pd.Series, "sharpe": float, "cagr": float, "mdd": float, "total_execution_cost": float}
    config : CapacityConfig
        Capital levels and execution parameters.
    """

    def __init__(self, simulation_fn: Callable, config: CapacityConfig) -> None:
        self.sim_fn = simulation_fn
        self.config = config
        self.slippage = SlippageModel(fixed_bps=config.fixed_slippage_bps)
        self.impact = SquareRootImpactModel(impact_coefficient=config.impact_coefficient)
        self.liquidity = LiquidityModel(max_participation_rate=config.max_participation)
        self._results: dict[float, dict] = {}

    def run_capacity_analysis(self) -> pd.DataFrame:
        """Run the simulation across all capital levels.

        Returns
        -------
        pd.DataFrame
            Summary of metrics per capital level.
        """
        results = []
        
        # Baseline (small capital, no impact)
        base_cap = self.config.capital_levels[0]
        logger.info("Running baseline capacity analysis at $%s", f"{base_cap:,.0f}")
        
        for cap in self.config.capital_levels:
            logger.info("Simulating execution at capital level $%s", f"{cap:,.0f}")
            res = self.sim_fn(cap, self.slippage, self.impact, self.liquidity)
            self._results[cap] = res
            
            # Extract key metrics
            sharpe = res.get("sharpe", np.nan)
            cagr = res.get("cagr", np.nan)
            mdd = res.get("mdd", np.nan)
            cost = res.get("total_execution_cost", 0.0)
            
            results.append({
                "capital": cap,
                "sharpe": round(sharpe, 4),
                "cagr": round(cagr, 6),
                "max_drawdown": round(mdd, 4),
                "execution_cost": round(cost, 2),
                "cost_bps": round((cost / cap) * 10000.0, 2) if cap > 0 else 0.0,
            })
            
        return pd.DataFrame(results).set_index("capital")

    def detect_capacity_breakpoint(self, sharpe_decay_threshold: float = 0.5) -> float:
        """Identify the capital level where Sharpe ratio decays by a certain threshold.

        Returns
        -------
        float
            Maximum recommended deployable capital.
        """
        if not self._results:
            self.run_capacity_analysis()
            
        caps = sorted(list(self._results.keys()))
        if not caps:
            return 0.0
            
        base_sharpe = self._results[caps[0]].get("sharpe", 0.0)
        if base_sharpe <= 0:
            return caps[0]
            
        threshold = base_sharpe * sharpe_decay_threshold
        
        for cap in caps:
            s = self._results[cap].get("sharpe", 0.0)
            if s < threshold:
                return cap
                
        return caps[-1]  # No breakdown within tested levels
