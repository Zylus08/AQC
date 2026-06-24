"""
aqc/research/features/feature_generator.py
============================================
Orchestrates generation of all features for a given dataset.

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging

import pandas as pd

from aqc.research.features.feature_library import FeatureLibrary

logger = logging.getLogger(__name__)


class FeatureGenerator:
    """Generates a complete feature matrix."""

    def __init__(self) -> None:
        self.lib = FeatureLibrary()

    def generate(self, trades: pd.DataFrame, snapshots: pd.DataFrame) -> pd.DataFrame:
        """Combine all features into a single matrix.
        
        Parameters
        ----------
        trades:
            Raw trade data.
        snapshots:
            Raw order book data.
            
        Returns
        -------
        pd.DataFrame
            Feature matrix indexed by snapshot timestamp.
        """
        logger.info("Generating feature matrix...")
        
        ob_feats = self.lib.get_orderbook_features(snapshots)
        of_feats = self.lib.get_orderflow_features(trades, snapshots)
        
        combined = pd.concat([ob_feats, of_feats], axis=1).ffill().fillna(0.0)
        
        logger.info("Generated %d features.", len(combined.columns))
        return combined
