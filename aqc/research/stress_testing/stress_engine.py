"""
aqc/research/stress_testing/stress_engine.py
==============================================
Orchestrates the complete stress testing suite.

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from aqc.alpha.alpha_base import AlphaBase
from aqc.research.stress_testing.parameter_stability import ParameterStabilityTester
from aqc.research.stress_testing.monte_carlo import MonteCarloSimulator

logger = logging.getLogger(__name__)


class StressTestEngine:
    """Runs all stress tests on an alpha."""

    def __init__(self) -> None:
        self.param_tester = ParameterStabilityTester()
        self.mc_simulator = MonteCarloSimulator()

    def run_full_suite(
        self, 
        alpha: AlphaBase, 
        X: pd.DataFrame, 
        y: pd.Series, 
        params_to_test: list[str] = None
    ) -> dict[str, Any]:
        """Run all stress tests.

        Returns
        -------
        dict
            Results containing parameter stability data and monte carlo CIs.
        """
        logger.info("Starting Stress Test Suite for %s", alpha.name)
        results = {}

        # 1. Parameter Stability
        param_results = {}
        if params_to_test:
            for p in params_to_test:
                df = self.param_tester.run(alpha, p, X, y)
                if not df.empty:
                    param_results[p] = df
        results["parameter_stability"] = param_results

        # 2. Monte Carlo
        preds = alpha.predict(X)
        mc_df = self.mc_simulator.run(alpha, preds, y)
        results["monte_carlo"] = mc_df

        # 3. Aggregate Summary
        if not mc_df.empty:
            summary = {
                "Sharpe_5th": mc_df["Sharpe"].quantile(0.05),
                "Sharpe_50th": mc_df["Sharpe"].median(),
                "Sharpe_95th": mc_df["Sharpe"].quantile(0.95),
                "Max_DD_95th": mc_df["Max_DD"].quantile(0.95), # Worst case DD (assuming positive is loss)
            }
            results["summary"] = summary
            
            # Stress Test Confidence Score
            # Ratio of 5th percentile Sharpe to Median Sharpe
            if summary["Sharpe_50th"] > 0:
                stress_score = max(0.0, summary["Sharpe_5th"] / summary["Sharpe_50th"])
            else:
                stress_score = 0.0
            results["stress_score"] = stress_score
            logger.info("Stress Score for %s: %.2f", alpha.name, stress_score)

        return results
