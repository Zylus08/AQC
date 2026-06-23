"""
aqc/strategies/microprice_alpha/microprice_features.py
========================================================
Feature extraction for Microprice Alpha.

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from aqc.orderbook.microprice import MicropriceEstimator

logger = logging.getLogger(__name__)


class MicropriceFeatureEngine:
    """Compute microprice, deviation, and fair value gap.

    Examples
    --------
    >>> engine = MicropriceFeatureEngine()
    >>> features = engine.extract(snapshots_df)
    """

    def __init__(self, n_levels: int = 5, zscore_window: int = 20) -> None:
        self.n_levels = n_levels
        self.zscore_window = zscore_window
        self._microprice = MicropriceEstimator()

    def extract(self, snapshots: pd.DataFrame) -> pd.DataFrame:
        """Extract microprice features from snapshot data.

        Parameters
        ----------
        snapshots:
            Order book snapshots.

        Returns
        -------
        pd.DataFrame
        """
        results: dict[str, list[float]] = {
            "microprice": [],
            "microprice_deviation": [],
            "fair_value_gap": [],
        }

        for _, row in snapshots.iterrows():
            bb = float(row.get("bid_px_0", 0.0))
            bv = float(row.get("bid_vol_0", 0.0))
            ba = float(row.get("ask_px_0", 0.0))
            av = float(row.get("ask_vol_0", 0.0))
            mid = float(row.get("mid_price", (bb + ba) / 2))

            mp = self._microprice.calculate_microprice(bb, bv, ba, av)
            results["microprice"].append(mp)
            results["microprice_deviation"].append(mp - mid)

            bids = self._extract_levels(row, "bid")
            asks = self._extract_levels(row, "ask")
            fv = self._microprice.calculate_fair_value(bids, asks, self.n_levels)
            results["fair_value_gap"].append(fv - mid)

        df = pd.DataFrame(results, index=snapshots.index)

        # Rolling z-score of deviation
        roll_mean = df["microprice_deviation"].rolling(self.zscore_window).mean()
        roll_std = df["microprice_deviation"].rolling(self.zscore_window).std()
        df["deviation_zscore"] = (df["microprice_deviation"] - roll_mean) / roll_std.replace(0, 1)

        # Fair value z-score
        fv_mean = df["fair_value_gap"].rolling(self.zscore_window).mean()
        fv_std = df["fair_value_gap"].rolling(self.zscore_window).std()
        df["fair_value_zscore"] = (df["fair_value_gap"] - fv_mean) / fv_std.replace(0, 1)

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
