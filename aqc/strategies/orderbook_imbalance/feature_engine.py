"""
aqc/strategies/orderbook_imbalance/feature_engine.py
======================================================
Feature engineering pipeline for order book imbalance alpha.

Delegates to existing AQC primitives:
- ``aqc.orderbook.ImbalanceEngine`` for volume imbalances
- ``aqc.orderbook.MicropriceEstimator`` for microprice
- ``aqc.orderbook.OrderbookFeatures`` for spread/depth/pressure
- ``aqc.microstructure.FlowToxicity`` for VPIN
- ``aqc.microstructure.LiquidityRegimes`` for regime labels

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from aqc.orderbook.imbalance_engine import ImbalanceEngine
from aqc.orderbook.microprice import MicropriceEstimator
from aqc.orderbook.orderbook_features import OrderbookFeatures
from aqc.microstructure.flow_toxicity import FlowToxicity
from aqc.microstructure.liquidity_regimes import LiquidityRegimes

logger = logging.getLogger(__name__)


class ImbalanceFeatureEngine:
    """Extract order book imbalance features from snapshot data.

    Wraps the existing AQC order book primitives into a single
    feature-engineering pipeline that produces a feature matrix
    ready for ML model training.

    Parameters
    ----------
    n_levels:
        Number of price levels available in the snapshot data.
    vpin_bucket_size:
        Volume per VPIN bucket.

    Examples
    --------
    >>> engine = ImbalanceFeatureEngine(n_levels=10)
    >>> features = engine.extract(snapshots_df, trades_df)
    """

    def __init__(
        self,
        n_levels: int = 10,
        vpin_bucket_size: float = 5000.0,
    ) -> None:
        self.n_levels = n_levels
        self.vpin_bucket_size = vpin_bucket_size

        # Delegate to existing AQC modules
        self._imbalance = ImbalanceEngine()
        self._microprice = MicropriceEstimator()
        self._ob_features = OrderbookFeatures()
        self._flow_toxicity = FlowToxicity()
        self._liquidity_regimes = LiquidityRegimes()

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def extract(
        self,
        snapshots: pd.DataFrame,
        trades: Optional[pd.DataFrame] = None,
        events: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Extract full feature set from order book data.

        Parameters
        ----------
        snapshots:
            Order book snapshots with columns ``bid_px_0..N``,
            ``bid_vol_0..N``, ``ask_px_0..N``, ``ask_vol_0..N``,
            ``mid_price``, ``spread``.
        trades:
            Trade data with columns ``price``, ``volume``, ``direction``.
        events:
            Order book events with ``event_type``, ``side``, ``price``,
            ``volume`` (optional).

        Returns
        -------
        pd.DataFrame
            Feature matrix indexed by timestamp.
        """
        features = pd.DataFrame(index=snapshots.index)

        # 1. Volume imbalances at various depth levels
        imb_features = self._compute_imbalances(snapshots)
        features = features.join(imb_features)

        # 2. Microprice and fair value
        micro_features = self._compute_microprice_features(snapshots)
        features = features.join(micro_features)

        # 3. Spread and depth features
        sd_features = self._compute_spread_depth(snapshots)
        features = features.join(sd_features)

        # 4. Queue imbalance (change in depth)
        queue_features = self._compute_queue_imbalance(snapshots)
        features = features.join(queue_features)

        # 5. Trade flow features (if trades available)
        if trades is not None and not trades.empty:
            flow_features = self._compute_trade_flow(snapshots, trades)
            features = features.join(flow_features)

        # 6. Order event features (arrivals, cancellations)
        if events is not None and not events.empty:
            event_features = self._compute_event_features(snapshots, events)
            features = features.join(event_features)

        # 7. Liquidity regime label
        regime_features = self._compute_liquidity_regime(features)
        features = features.join(regime_features)

        return features

    # ------------------------------------------------------------------
    # Prediction targets
    # ------------------------------------------------------------------

    def compute_targets(
        self,
        snapshots: pd.DataFrame,
        horizons: tuple[int, ...] = (1, 5, 10),
    ) -> pd.DataFrame:
        """Compute prediction targets (future mid-price direction).

        Parameters
        ----------
        snapshots:
            Snapshot data with ``mid_price`` column.
        horizons:
            Forward-looking horizons in number of snapshots.

        Returns
        -------
        pd.DataFrame
            Columns: ``ret_N``, ``dir_N`` for each horizon N.
        """
        targets = pd.DataFrame(index=snapshots.index)
        mid = snapshots["mid_price"]

        for h in horizons:
            future_ret = mid.shift(-h) / mid - 1
            targets[f"ret_{h}"] = future_ret
            targets[f"dir_{h}"] = np.sign(future_ret).astype(int)

        return targets

    # ------------------------------------------------------------------
    # Feature groups
    # ------------------------------------------------------------------

    def _compute_imbalances(self, snapshots: pd.DataFrame) -> pd.DataFrame:
        """Compute volume imbalances at various depth levels."""
        results: dict[str, list[float]] = {
            "imb_top1": [], "imb_top5": [], "imb_top10": [],
            "depth_imbalance": [],
        }

        for _, row in snapshots.iterrows():
            bids = self._extract_levels(row, "bid")
            asks = self._extract_levels(row, "ask")

            results["imb_top1"].append(
                self._imbalance.compute_imbalance(bids, asks, 1)
            )
            results["imb_top5"].append(
                self._imbalance.compute_imbalance(bids, asks, 5)
            )
            results["imb_top10"].append(
                self._imbalance.compute_imbalance(bids, asks, min(10, len(bids)))
            )

            # Depth imbalance: weighted by inverse distance from mid
            bid_depth = sum(v for _, v in bids)
            ask_depth = sum(v for _, v in asks)
            total = bid_depth + ask_depth
            results["depth_imbalance"].append(
                (bid_depth - ask_depth) / total if total > 0 else 0.0
            )

        return pd.DataFrame(results, index=snapshots.index)

    def _compute_microprice_features(self, snapshots: pd.DataFrame) -> pd.DataFrame:
        """Compute microprice and related signals."""
        results: dict[str, list[float]] = {
            "microprice": [],
            "microprice_deviation": [],
            "fair_value_gap": [],
        }

        for _, row in snapshots.iterrows():
            bb = row.get("bid_px_0", 0.0)
            bv = row.get("bid_vol_0", 0.0)
            ba = row.get("ask_px_0", 0.0)
            av = row.get("ask_vol_0", 0.0)
            mid = row.get("mid_price", (bb + ba) / 2)

            mp = self._microprice.calculate_microprice(bb, bv, ba, av)
            results["microprice"].append(mp)
            results["microprice_deviation"].append(mp - mid)

            bids = self._extract_levels(row, "bid")
            asks = self._extract_levels(row, "ask")
            fv = self._microprice.calculate_fair_value(bids, asks, 5)
            results["fair_value_gap"].append(fv - mid)

        return pd.DataFrame(results, index=snapshots.index)

    def _compute_spread_depth(self, snapshots: pd.DataFrame) -> pd.DataFrame:
        """Compute spread, depth, and pressure features."""
        results: list[dict] = []

        for _, row in snapshots.iterrows():
            bids = self._extract_levels(row, "bid")
            asks = self._extract_levels(row, "ask")
            feats = self._ob_features.compute_features(bids, asks)
            results.append(feats)

        return pd.DataFrame(results, index=snapshots.index)

    def _compute_queue_imbalance(self, snapshots: pd.DataFrame) -> pd.DataFrame:
        """Compute change in depth at best bid/ask (queue imbalance)."""
        bid_vol = snapshots.get("bid_vol_0", pd.Series(0, index=snapshots.index))
        ask_vol = snapshots.get("ask_vol_0", pd.Series(0, index=snapshots.index))

        delta_bid = bid_vol.diff().fillna(0)
        delta_ask = ask_vol.diff().fillna(0)
        total_delta = delta_bid.abs() + delta_ask.abs()

        queue_imb = (delta_bid - delta_ask) / total_delta.replace(0, 1)

        return pd.DataFrame({
            "queue_imbalance": queue_imb,
            "delta_bid_vol": delta_bid,
            "delta_ask_vol": delta_ask,
        }, index=snapshots.index)

    def _compute_trade_flow(
        self,
        snapshots: pd.DataFrame,
        trades: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute trade flow features aggregated per snapshot."""
        results: dict[str, list[float]] = {
            "buy_volume": [],
            "sell_volume": [],
            "net_flow": [],
            "trade_intensity": [],
            "flow_imbalance": [],
        }

        for ts in snapshots.index:
            # Get trades at or near this timestamp
            mask = trades.index == ts
            snapshot_trades = trades[mask]

            buy_vol = float(
                snapshot_trades[snapshot_trades["direction"] == "BUY"]["volume"].sum()
            )
            sell_vol = float(
                snapshot_trades[snapshot_trades["direction"] == "SELL"]["volume"].sum()
            )
            total = buy_vol + sell_vol

            results["buy_volume"].append(buy_vol)
            results["sell_volume"].append(sell_vol)
            results["net_flow"].append(buy_vol - sell_vol)
            results["trade_intensity"].append(len(snapshot_trades))
            results["flow_imbalance"].append(
                (buy_vol - sell_vol) / total if total > 0 else 0.0
            )

        return pd.DataFrame(results, index=snapshots.index)

    def _compute_event_features(
        self,
        snapshots: pd.DataFrame,
        events: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute order arrival and cancellation features."""
        results: dict[str, list[float]] = {
            "arrival_imbalance": [],
            "cancellation_imbalance": [],
            "modification_rate": [],
        }

        for ts in snapshots.index:
            mask = events.index == ts
            ts_events = events[mask]

            arrivals = ts_events[ts_events["event_type"] == "ARRIVAL"]
            cancellations = ts_events[ts_events["event_type"] == "CANCELLATION"]
            modifications = ts_events[ts_events["event_type"] == "MODIFICATION"]

            # Arrival imbalance
            arr_bid = arrivals[arrivals["side"] == "BID"]["volume"].sum()
            arr_ask = arrivals[arrivals["side"] == "ASK"]["volume"].sum()
            arr_total = arr_bid + arr_ask
            results["arrival_imbalance"].append(
                (arr_bid - arr_ask) / arr_total if arr_total > 0 else 0.0
            )

            # Cancellation imbalance
            can_bid = cancellations[cancellations["side"] == "BID"]["volume"].sum()
            can_ask = cancellations[cancellations["side"] == "ASK"]["volume"].sum()
            can_total = can_bid + can_ask
            results["cancellation_imbalance"].append(
                (can_bid - can_ask) / can_total if can_total > 0 else 0.0
            )

            # Modification rate
            total_events = len(ts_events)
            results["modification_rate"].append(
                len(modifications) / total_events if total_events > 0 else 0.0
            )

        return pd.DataFrame(results, index=snapshots.index)

    def _compute_liquidity_regime(self, features: pd.DataFrame) -> pd.DataFrame:
        """Label each snapshot with a liquidity regime."""
        regimes = []
        for _, row in features.iterrows():
            spread = row.get("spread", 0.01)
            depth = row.get("total_depth", 10000)
            impact = row.get("flow_imbalance", 0.0)
            regime = self._liquidity_regimes.detect_regime(
                float(spread), float(depth), abs(float(impact))
            )
            regimes.append(regime)

        return pd.DataFrame({"liquidity_regime": regimes}, index=features.index)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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
