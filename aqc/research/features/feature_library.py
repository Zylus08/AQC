"""
aqc/research/features/feature_library.py
==========================================
Unified library of all AQC feature extractors.

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging
from typing import Callable

import pandas as pd

from aqc.orderbook.imbalance_engine import ImbalanceEngine
from aqc.orderbook.microprice import MicropriceEstimator
from aqc.orderbook.orderbook_features import OrderbookFeatures
from aqc.microstructure.flow_toxicity import FlowToxicity
from aqc.microstructure.trade_signing import TradeSigner

logger = logging.getLogger(__name__)


class FeatureLibrary:
    """Registry of core feature extraction primitives."""

    def __init__(self) -> None:
        self.ob_engine = ImbalanceEngine()
        self.mp_engine = MicropriceEstimator()
        self.of_features = OrderbookFeatures()
        self.toxicity = FlowToxicity()
        self.signer = TradeSigner()

    def get_orderbook_features(self, snapshots: pd.DataFrame) -> pd.DataFrame:
        """Extract all OB-related features."""
        # Simplified: loop through snapshots and compute
        # In production this would be vectorized or map-reduced
        results = []
        for _, row in snapshots.iterrows():
            # For simplicity in the generalized feature engine, 
            # we assume `bid_px_0`, `bid_vol_0`, etc.
            try:
                b0 = float(row.get("bid_vol_0", 0))
                a0 = float(row.get("ask_vol_0", 0))
                imb = self.ob_engine.calculate_imbalance([b0], [a0], depth=1)
                
                bids = [(float(row.get(f"bid_px_{i}", 0)), float(row.get(f"bid_vol_{i}", 0))) for i in range(5)]
                asks = [(float(row.get(f"ask_px_{i}", 0)), float(row.get(f"ask_vol_{i}", 0))) for i in range(5)]
                
                feats = self.of_features.compute_features(bids, asks)
                feats["ob_imbalance"] = imb
                results.append(feats)
            except Exception:
                pass
                
        return pd.DataFrame(results, index=snapshots.index)

    def get_orderflow_features(self, trades: pd.DataFrame, snapshots: pd.DataFrame) -> pd.DataFrame:
        """Extract all OF-related features (VPIN, signed flow)."""
        if "direction" not in trades.columns:
            trades = self.signer.sign_trades(trades, snapshots)
            
        trades["signed_volume"] = trades["volume"] * trades["direction"].map({"BUY": 1, "SELL": -1})
        vpin = self.toxicity.calculate_vpin(trades)
        
        # Align to snapshot index
        df = pd.DataFrame(index=snapshots.index)
        df["vpin"] = vpin.reindex(df.index, method="ffill").fillna(0.0)
        return df
