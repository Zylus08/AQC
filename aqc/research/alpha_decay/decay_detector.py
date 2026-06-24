"""
aqc/research/alpha_decay/decay_detector.py
============================================
Detects structural alpha decay.

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DecayDetector:
    """Flags alphas that show structural decay in IC or Return."""

    def __init__(self, ic_drop_threshold: float = 0.5, halflife_threshold_bars: int = 500) -> None:
        self.ic_drop_threshold = ic_drop_threshold
        self.halflife_threshold = halflife_threshold_bars

    def detect(self, alpha_name: str, rolling_ic: pd.Series) -> dict:
        """Analyze a rolling IC series for decay.

        Returns
        -------
        dict
            Decay analysis containing boolean flag `is_decayed`.
        """
        if len(rolling_ic) < 100:
            return {"is_decayed": False, "reason": "Insufficient data"}

        # Recent IC vs Historical IC
        recent_ic = rolling_ic.tail(50).mean()
        historical_ic = rolling_ic.iloc[:-50].mean()

        # Check for severe drop
        drop_pct = 0.0
        if historical_ic > 0:
            drop_pct = (historical_ic - recent_ic) / historical_ic

        is_decayed = False
        reason = ""

        if drop_pct > self.ic_drop_threshold:
            is_decayed = True
            reason = f"IC dropped by {drop_pct*100:.1f}% vs historical baseline."
        elif recent_ic < 0.0:
            is_decayed = True
            reason = "Recent IC is negative."

        # Estimate Half-life (simplified proxy: bars until IC hits 50% of peak)
        peak_ic = rolling_ic.max()
        peak_idx = rolling_ic.argmax()
        
        # Look forward from peak
        post_peak = rolling_ic.iloc[peak_idx:]
        halflife_bars = -1
        
        # Find first index where IC < peak_ic / 2
        below_half = post_peak[post_peak < (peak_ic / 2)]
        if not below_half.empty:
            halflife_bars = len(post_peak.loc[:below_half.index[0]])
            
            if halflife_bars < self.halflife_threshold:
                is_decayed = True
                reason = f"Fast decay half-life: {halflife_bars} bars."

        return {
            "alpha_name": alpha_name,
            "is_decayed": is_decayed,
            "reason": reason,
            "recent_ic": recent_ic,
            "historical_ic": historical_ic,
            "halflife_bars": halflife_bars,
        }
