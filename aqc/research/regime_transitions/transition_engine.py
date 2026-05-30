"""
aqc/research/regime_transitions/transition_engine.py
======================================================
Identify and record all market regime transitions.

Author: Saksham Mishra — AlgoQuant Club
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TransitionEvent:
    """A change in market regime."""
    transition_id: int
    timestamp: pd.Timestamp
    regime_type: str         # "volatility" or "trend"
    from_regime: str
    to_regime: str
    price_at_transition: float


class TransitionEngine:
    """Identify regime transitions from a regime history.

    Parameters
    ----------
    regime_data : pd.DataFrame
        Must contain columns 'vol_regime' and 'trend_regime'.
    prices : pd.Series
        Price series to capture price at transition.
    """

    def __init__(self, regime_data: pd.DataFrame, prices: pd.Series) -> None:
        self.regime_data = regime_data.dropna(subset=["vol_regime", "trend_regime"])
        self.prices = prices
        self._events: list[TransitionEvent] = []

    def identify_transitions(self) -> list[TransitionEvent]:
        """Scan series and record state changes."""
        events = []
        event_id = 0

        # Align
        common_idx = self.regime_data.index.intersection(self.prices.index)
        rd = self.regime_data.loc[common_idx]
        px = self.prices.loc[common_idx]

        if len(rd) < 2:
            return events

        prev_vol = str(rd["vol_regime"].iloc[0])
        prev_trend = str(rd["trend_regime"].iloc[0])

        for i in range(1, len(rd)):
            ts = rd.index[i]
            cur_vol = str(rd["vol_regime"].iloc[i])
            cur_trend = str(rd["trend_regime"].iloc[i])
            price = float(px.iloc[i])

            if cur_vol != prev_vol:
                event_id += 1
                events.append(
                    TransitionEvent(
                        transition_id=event_id,
                        timestamp=ts,
                        regime_type="volatility",
                        from_regime=prev_vol,
                        to_regime=cur_vol,
                        price_at_transition=price,
                    )
                )
                prev_vol = cur_vol

            if cur_trend != prev_trend:
                event_id += 1
                events.append(
                    TransitionEvent(
                        transition_id=event_id,
                        timestamp=ts,
                        regime_type="trend",
                        from_regime=prev_trend,
                        to_regime=cur_trend,
                        price_at_transition=price,
                    )
                )
                prev_trend = cur_trend

        self._events = events
        logger.info("Identified %d regime transitions.", len(events))
        return events

    def get_events_df(self) -> pd.DataFrame:
        events = self._events if self._events else self.identify_transitions()
        if not events:
            return pd.DataFrame()
        return pd.DataFrame([e.__dict__ for e in events])
