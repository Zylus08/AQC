"""
aqc/strategies/orderflow_alpha/orderflow_features.py
======================================================
Feature extraction for Order Flow Alpha.

Delegates to existing AQC primitives:
- `aqc.microstructure.OrderFlow`
- `aqc.microstructure.TradeSigner`
- `aqc.microstructure.FlowToxicity`

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging

import pandas as pd

from aqc.microstructure.order_flow import OrderFlow
from aqc.microstructure.trade_signing import TradeSigner
from aqc.microstructure.flow_toxicity import FlowToxicity

logger = logging.getLogger(__name__)


class OrderFlowFeatureEngine:
    """Extract order flow features from trade and snapshot data.

    Computes signed volume, trade intensity, VPIN, and flow imbalance.
    """

    def __init__(self, bucket_volume: float = 50000.0) -> None:
        self.bucket_volume = bucket_volume
        self._order_flow = OrderFlow()
        self._trade_signer = TradeSigner()
        self._flow_toxicity = FlowToxicity(bucket_volume=bucket_volume)

    def extract(self, trades: pd.DataFrame, snapshots: pd.DataFrame) -> pd.DataFrame:
        """Extract order flow features.

        Parameters
        ----------
        trades:
            DataFrame with `price`, `volume`, and optionally `direction`.
            If `direction` is missing, it will be inferred.
        snapshots:
            DataFrame with `mid_price` used for trade signing.

        Returns
        -------
        pd.DataFrame
            Feature matrix aggregated to snapshot timestamps.
        """
        if trades.empty or snapshots.empty:
            return pd.DataFrame()

        # Sign trades if necessary
        if "direction" not in trades.columns:
            trades = self._trade_signer.sign_trades(trades, snapshots)

        results: dict[str, list[float]] = {
            "buy_volume": [],
            "sell_volume": [],
            "net_flow": [],
            "flow_imbalance": [],
            "trade_count": [],
        }

        # VPIN requires a continuous stream of signed volume, we compute it globally
        trades["signed_volume"] = trades["volume"] * trades["direction"].map({"BUY": 1, "SELL": -1})
        vpin_series = self._flow_toxicity.calculate_vpin(trades)

        for i, ts in enumerate(snapshots.index):
            # Aggregation window: from previous snapshot to current
            prev_ts = snapshots.index[i-1] if i > 0 else trades.index[0]
            
            mask = (trades.index > prev_ts) & (trades.index <= ts)
            window_trades = trades[mask]

            buy_vol = window_trades[window_trades["direction"] == "BUY"]["volume"].sum()
            sell_vol = window_trades[window_trades["direction"] == "SELL"]["volume"].sum()
            total = buy_vol + sell_vol

            results["buy_volume"].append(float(buy_vol))
            results["sell_volume"].append(float(sell_vol))
            results["net_flow"].append(float(buy_vol - sell_vol))
            results["flow_imbalance"].append(
                float((buy_vol - sell_vol) / total) if total > 0 else 0.0
            )
            results["trade_count"].append(float(len(window_trades)))

        df = pd.DataFrame(results, index=snapshots.index)
        
        # Align VPIN (which is indexed by trade buckets or trade time) to snapshots
        if not vpin_series.empty:
            df["vpin"] = vpin_series.reindex(df.index, method="ffill").fillna(0.0)
        else:
            df["vpin"] = 0.0

        # Rolling smoothed metrics
        df["flow_imbalance_smooth"] = df["flow_imbalance"].ewm(span=10).mean()
        
        return df
