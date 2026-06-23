"""
aqc/strategies/liquidity_alpha/liquidity_features.py
======================================================
Feature extraction for Liquidity Alpha.

Delegates to existing AQC primitives:
- `aqc.microstructure.LiquidityRegimes`
- `aqc.orderbook.OrderbookFeatures`

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging

import pandas as pd

from aqc.microstructure.liquidity_regimes import LiquidityRegimes
from aqc.orderbook.orderbook_features import OrderbookFeatures

logger = logging.getLogger(__name__)


class LiquidityFeatureEngine:
    """Extract liquidity features from order book snapshots.

    Computes spread expansion/compression, depth shocks,
    and liquidity withdrawal rates.
    """

    def __init__(self, n_levels: int = 5, lookback: int = 20) -> None:
        self.n_levels = n_levels
        self.lookback = lookback
        self._regimes = LiquidityRegimes()
        self._ob_features = OrderbookFeatures()

    def extract(self, snapshots: pd.DataFrame) -> pd.DataFrame:
        """Extract liquidity features from snapshot data."""
        results = []

        for _, row in snapshots.iterrows():
            bids = self._extract_levels(row, "bid")
            asks = self._extract_levels(row, "ask")

            # Spread and Depth
            feats = self._ob_features.compute_features(bids, asks)
            
            # Regime label mapping string to integer for ML
            impact = 0.0  # Optional impact override if known
            regime = self._regimes.detect_regime(
                spread=feats.get("spread", 0.01),
                depth=feats.get("total_depth", 1000.0),
                impact=impact,
            )
            
            # Map regime to simple categorical int
            # Regimes output string enum like "NORMAL", "STRESSED", "SHOCK"
            reg_map = {"NORMAL": 0, "STRESSED": 1, "SHOCK": 2}
            reg_val = reg_map.get(regime, 0)

            feats["liquidity_regime"] = reg_val
            results.append(feats)

        df = pd.DataFrame(results, index=snapshots.index)

        # Rolling Liquidity Metrics
        if "spread" in df.columns:
            roll_mean = df["spread"].rolling(self.lookback).mean()
            roll_std = df["spread"].rolling(self.lookback).std()
            df["spread_zscore"] = (df["spread"] - roll_mean) / roll_std.replace(0, 1)

        if "total_depth" in df.columns:
            roll_mean = df["total_depth"].rolling(self.lookback).mean()
            roll_std = df["total_depth"].rolling(self.lookback).std()
            df["depth_zscore"] = (df["total_depth"] - roll_mean) / roll_std.replace(0, 1)
            
            # Liquidity shock: sudden drop in depth
            df["depth_drop"] = df["total_depth"].diff().clip(upper=0).abs()
            df["is_liquidity_shock"] = (df["depth_zscore"] < -2.0).astype(int)

        return df

    def _extract_levels(
        self, row: pd.Series, side: str
    ) -> list[tuple[float, float]]:
        """Extract (price, volume) tuples from a snapshot row."""
        levels = []
        for i in range(self.n_levels):
            px_col = f"{side}_px_{i}"
            vol_col = f"{side}_vol_{i}"
            if px_col in row.index and vol_col in row.index:
                px = float(row[px_col])
                vol = float(row[vol_col])
                if px > 0 and vol > 0:
                    levels.append((px, vol))
        return levels
