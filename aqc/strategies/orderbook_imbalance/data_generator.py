"""
aqc/strategies/orderbook_imbalance/data_generator.py
======================================================
Synthetic L2 order book data generator for alpha research.

Generates realistic bid/ask ladders with configurable depth,
trade arrivals, cancellations, and signed order flow suitable
for feature engineering and model training.

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SyntheticOrderBookGenerator:
    """Generate synthetic L2 order book snapshots and trade data.

    Parameters
    ----------
    n_snapshots:
        Number of order book snapshots to generate.
    n_levels:
        Number of price levels per side.
    base_price:
        Starting mid-price.
    tick_size:
        Minimum price increment.
    base_volume:
        Mean volume per level.
    volatility:
        Price random walk volatility.
    seed:
        Random seed for reproducibility.

    Examples
    --------
    >>> gen = SyntheticOrderBookGenerator(n_snapshots=1000, seed=42)
    >>> snapshots, trades = gen.generate()
    """

    def __init__(
        self,
        n_snapshots: int = 5000,
        n_levels: int = 10,
        base_price: float = 100.0,
        tick_size: float = 0.01,
        base_volume: float = 500.0,
        volatility: float = 0.001,
        seed: int = 42,
    ) -> None:
        self.n_snapshots = n_snapshots
        self.n_levels = n_levels
        self.base_price = base_price
        self.tick_size = tick_size
        self.base_volume = base_volume
        self.volatility = volatility
        self.rng = np.random.RandomState(seed)

    def generate(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Generate synthetic order book snapshots and trade data.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            (snapshots_df, trades_df)

            ``snapshots_df`` columns per level:
            ``bid_px_0..N``, ``bid_vol_0..N``,
            ``ask_px_0..N``, ``ask_vol_0..N``,
            ``mid_price``, ``spread``, ``timestamp``

            ``trades_df`` columns:
            ``timestamp``, ``price``, ``volume``, ``direction``
        """
        timestamps = pd.date_range(
            "2024-01-02 09:30:00",
            periods=self.n_snapshots,
            freq="100ms",
        )

        mid_prices = self._generate_mid_prices()
        snapshots = []
        trades = []

        for i in range(self.n_snapshots):
            mid = mid_prices[i]
            spread = self.tick_size * (1 + self.rng.exponential(0.5))
            best_bid = mid - spread / 2
            best_ask = mid + spread / 2

            row: dict = {
                "timestamp": timestamps[i],
                "mid_price": round(mid, 4),
                "spread": round(spread, 6),
            }

            # Generate bid levels
            for level in range(self.n_levels):
                px = best_bid - level * self.tick_size
                vol = self.base_volume * self.rng.exponential(1.0) * (
                    1.0 + 0.3 * level
                )
                row[f"bid_px_{level}"] = round(px, 4)
                row[f"bid_vol_{level}"] = round(vol, 0)

            # Generate ask levels
            for level in range(self.n_levels):
                px = best_ask + level * self.tick_size
                vol = self.base_volume * self.rng.exponential(1.0) * (
                    1.0 + 0.3 * level
                )
                row[f"ask_px_{level}"] = round(px, 4)
                row[f"ask_vol_{level}"] = round(vol, 0)

            snapshots.append(row)

            # Generate trades (Poisson arrival, ~2 per snapshot)
            n_trades = self.rng.poisson(2)
            for _ in range(n_trades):
                direction = "BUY" if self.rng.random() > 0.5 else "SELL"
                if direction == "BUY":
                    trade_px = best_ask + self.rng.exponential(0.002)
                else:
                    trade_px = best_bid - self.rng.exponential(0.002)
                trade_vol = self.rng.exponential(self.base_volume * 0.3)
                trades.append({
                    "timestamp": timestamps[i],
                    "price": round(trade_px, 4),
                    "volume": round(trade_vol, 0),
                    "direction": direction,
                })

        snapshots_df = pd.DataFrame(snapshots).set_index("timestamp")
        trades_df = pd.DataFrame(trades)
        if not trades_df.empty:
            trades_df = trades_df.set_index("timestamp")

        logger.info(
            "Generated %d snapshots, %d trades",
            len(snapshots_df), len(trades_df),
        )
        return snapshots_df, trades_df

    def _generate_mid_prices(self) -> np.ndarray:
        """Generate a random walk of mid-prices with mean reversion."""
        returns = self.rng.normal(0, self.volatility, self.n_snapshots)
        # Add slight mean-reversion
        log_prices = np.log(self.base_price) + np.cumsum(returns)
        mean_log = np.log(self.base_price)
        for i in range(1, len(log_prices)):
            log_prices[i] += 0.001 * (mean_log - log_prices[i - 1])
        return np.exp(log_prices)

    def generate_with_events(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generate snapshots, trades, and order book events.

        Order book events include arrivals, cancellations, and modifications.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            (snapshots, trades, events)
        """
        snapshots, trades = self.generate()

        events = []
        for ts in snapshots.index:
            n_events = self.rng.poisson(5)
            for _ in range(n_events):
                event_type = self.rng.choice(
                    ["ARRIVAL", "CANCELLATION", "MODIFICATION"],
                    p=[0.5, 0.3, 0.2],
                )
                side = self.rng.choice(["BID", "ASK"])
                mid = snapshots.loc[ts, "mid_price"]
                offset = self.rng.exponential(0.01)
                px = mid - offset if side == "BID" else mid + offset
                vol = self.rng.exponential(self.base_volume * 0.2)

                events.append({
                    "timestamp": ts,
                    "event_type": event_type,
                    "side": side,
                    "price": round(px, 4),
                    "volume": round(vol, 0),
                })

        events_df = pd.DataFrame(events)
        if not events_df.empty:
            events_df = events_df.set_index("timestamp")

        return snapshots, trades, events_df
