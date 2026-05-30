"""
aqc/execution/slippage_model.py
=================================
Models order slippage based on fixed bps, volatility, or spread.

Author: Saksham Mishra — AlgoQuant Club
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class SlippageModel:
    """Models execution slippage.

    Parameters
    ----------
    fixed_bps : float
        Fixed slippage in basis points (default: 0).
    volatility_multiplier : float
        Multiplier for daily volatility-based slippage (default: 0).
        slippage = vol_multiplier * daily_volatility
    """

    def __init__(self, fixed_bps: float = 0.0, volatility_multiplier: float = 0.0) -> None:
        self.fixed_bps = fixed_bps
        self.volatility_multiplier = volatility_multiplier

    def estimate_slippage(
        self, price: float, daily_volatility: Optional[float] = None
    ) -> float:
        """Estimate slippage in absolute price terms.

        Parameters
        ----------
        price : float
            Current market price.
        daily_volatility : float, optional
            Daily volatility (e.g., from ATR or returns std).

        Returns
        -------
        float
            Slippage amount (absolute price difference). Always positive.
        """
        slippage = 0.0

        if self.fixed_bps > 0:
            slippage += price * (self.fixed_bps / 10000.0)

        if self.volatility_multiplier > 0 and daily_volatility is not None:
            # Assume daily_volatility is proportional (e.g., 0.02 for 2%)
            slippage += price * daily_volatility * self.volatility_multiplier

        return slippage

    def apply_slippage(
        self, price: float, side: str, daily_volatility: Optional[float] = None
    ) -> float:
        """Apply slippage to an execution price.

        Parameters
        ----------
        price : float
            Base execution price.
        side : str
            "BUY" or "SELL".
        daily_volatility : float, optional
            Daily volatility constraint.

        Returns
        -------
        float
            Worse execution price.
        """
        slip = self.estimate_slippage(price, daily_volatility)
        if side == "BUY":
            return price + slip
        elif side == "SELL":
            return price - slip
        return price
