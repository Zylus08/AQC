"""
aqc/research/stress_testing/monte_carlo.py
============================================
Monte Carlo simulations for Alpha Stress Testing.

Randomizes execution timing, slippage, and spread to build
confidence intervals for performance metrics.

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging

import pandas as pd
import numpy as np

from aqc.alpha.alpha_base import AlphaBase

logger = logging.getLogger(__name__)


class MonteCarloSimulator:
    """Runs randomized simulations of alpha performance."""

    def __init__(self, iterations: int = 100, max_slippage_bps: float = 2.0) -> None:
        self.iterations = iterations
        self.max_slippage_bps = max_slippage_bps

    def run(self, alpha: AlphaBase, predictions: pd.Series, actuals: pd.Series) -> pd.DataFrame:
        """Run Monte Carlo simulation on predictions.

        Parameters
        ----------
        predictions:
            The base alpha predictions.
        actuals:
            The true target returns.
            
        Returns
        -------
        pd.DataFrame
            Metrics per iteration.
        """
        results = []
        
        # Base alignment
        aligned = pd.concat([predictions, actuals], axis=1).dropna()
        base_preds = aligned.iloc[:, 0]
        base_actuals = aligned.iloc[:, 1]
        
        logger.info("Running Monte Carlo simulation (%d iterations)...", self.iterations)

        for i in range(self.iterations):
            # 1. Randomize slippage
            # Slippage reduces the absolute return
            slippage = np.random.uniform(0, self.max_slippage_bps / 10000, size=len(base_actuals))
            
            # Simulated return: base return minus slippage cost per trade
            # Assume 1 trade per bar for simplicity in this proxy
            sim_actuals = base_actuals - slippage

            # 2. Randomize execution timing (signal delay)
            # 10% chance a signal is delayed by 1 bar
            delay_mask = np.random.rand(len(base_preds)) < 0.10
            sim_preds = base_preds.copy()
            delayed_preds = base_preds.shift(1).fillna(0)
            sim_preds[delay_mask] = delayed_preds[delay_mask]

            # 3. Evaluate
            metrics = alpha.evaluate(sim_preds, sim_actuals)
            
            results.append({
                "Iteration": i,
                "Sharpe": metrics.sharpe_ratio,
                "Sortino": metrics.sortino_ratio,
                "Max_DD": metrics.max_drawdown_pct,
            })

        df = pd.DataFrame(results)
        return df
