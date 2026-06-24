"""
aqc/research/generalization/market_validator.py
=================================================
Validates alpha performance across multiple market domains.

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import numpy as np

from aqc.alpha.alpha_base import AlphaBase

logger = logging.getLogger(__name__)


class SyntheticMarketDataGenerator:
    """Generates basic synthetic feature/target sets simulating different markets."""

    def __init__(self, n_samples: int = 1000):
        self.n_samples = n_samples

    def generate(self, market: str) -> tuple[pd.DataFrame, pd.Series]:
        """Generate synthetic (X, y) for a given market domain."""
        np.random.seed(hash(market) % (2**32))
        
        # Base noise
        features = pd.DataFrame({
            "feature_1": np.random.randn(self.n_samples),
            "feature_2": np.random.randn(self.n_samples),
        }, index=pd.date_range("2026-01-01", periods=self.n_samples, freq="1min"))
        
        # Target with varying signal-to-noise depending on market
        if market == "CRYPTO":
            # High vol, higher signal
            signal = features["feature_1"] * 0.5 + features["feature_2"] * 0.3
            noise = np.random.randn(self.n_samples) * 2.0
        elif market == "FX":
            # Low vol, mean reverting
            signal = -features["feature_1"] * 0.2 + features["feature_2"] * 0.1
            noise = np.random.randn(self.n_samples) * 0.5
        else: # EQUITIES
            signal = features["feature_1"] * 0.3 + features["feature_2"] * 0.2
            noise = np.random.randn(self.n_samples) * 1.0

        target = np.sign(signal + noise)
        return features, target


class CrossMarketValidator:
    """Runs an alpha across multiple market datasets."""

    def __init__(self, markets: list[str] = None) -> None:
        self.markets = markets or ["EQUITIES", "CRYPTO", "FX", "FUTURES"]
        self.data_gen = SyntheticMarketDataGenerator()

    def run(self, alpha: AlphaBase, market_data: dict[str, tuple[pd.DataFrame, pd.Series]] = None) -> dict[str, dict]:
        """Test alpha on multiple domains.
        
        If market_data is not provided, uses synthetic data.
        """
        results = {}
        
        for market in self.markets:
            logger.info("Validating %s on domain: %s", alpha.name, market)
            
            if market_data and market in market_data:
                X, y = market_data[market]
            else:
                X, y = self.data_gen.generate(market)
                
            try:
                preds = alpha.predict(X)
                metrics = alpha.evaluate(preds, y)
                results[market] = metrics.to_dict()
            except Exception as e:
                logger.error("Failed validation on %s: %s", market, e)
                results[market] = {"error": str(e)}

        return results
