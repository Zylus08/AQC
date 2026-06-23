"""
aqc/research/tournament/alpha_comparison.py
=============================================
Head-to-head Alpha Comparator.

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging

import pandas as pd

from aqc.alpha.alpha_base import AlphaBase

logger = logging.getLogger(__name__)


class AlphaComparator:
    """Compares two evaluated alphas head-to-head."""

    def __init__(self, alpha_a: AlphaBase, alpha_b: AlphaBase) -> None:
        if alpha_a.cached_metrics is None or alpha_b.cached_metrics is None:
            raise ValueError("Both alphas must be evaluated first.")
        self.alpha_a = alpha_a
        self.alpha_b = alpha_b

    def compare(self) -> pd.DataFrame:
        """Generate a side-by-side comparison DataFrame."""
        ma = self.alpha_a.cached_metrics.to_dict()
        mb = self.alpha_b.cached_metrics.to_dict()

        comparison = []
        for key in ma.keys():
            val_a = ma[key]
            val_b = mb[key]
            
            # Determine winner (lower is better for drawdown/turnover/decay)
            if key in ("max_drawdown_pct", "turnover", "decay_halflife_bars"):
                winner = self.alpha_a.name if val_a < val_b else (self.alpha_b.name if val_b < val_a else "TIE")
            else:
                winner = self.alpha_a.name if val_a > val_b else (self.alpha_b.name if val_b > val_a else "TIE")
                
            comparison.append({
                "Metric": key,
                self.alpha_a.name: val_a,
                self.alpha_b.name: val_b,
                "Winner": winner
            })

        return pd.DataFrame(comparison).set_index("Metric")
