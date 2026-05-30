"""
aqc/research/regime_transitions/transition_alpha.py
=====================================================
Calculate forward returns post-transition and evaluate statistical significance.

Author: Saksham Mishra — AlgoQuant Club
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class TransitionAlphaAnalyzer:
    """Analyze post-transition forward returns and compute statistical significance.

    Parameters
    ----------
    transitions_df : pd.DataFrame
        From TransitionEngine.get_events_df().
    prices : pd.Series
        Price series used to compute forward returns.
    horizons : list[int]
        Days forward to calculate returns (e.g., [1, 3, 5, 10, 20]).
    """

    def __init__(
        self,
        transitions_df: pd.DataFrame,
        prices: pd.Series,
        horizons: Optional[list[int]] = None,
    ) -> None:
        self.transitions = transitions_df.copy()
        self.prices = prices
        self.horizons = horizons or [1, 3, 5, 10, 20]

    def compute_forward_returns(self) -> pd.DataFrame:
        """Compute forward returns for each transition event."""
        if self.transitions.empty:
            return pd.DataFrame()

        df = self.transitions.copy()
        for h in self.horizons:
            # Shift prices backward to align future price with current index
            fut_px = self.prices.shift(-h)
            ret_series = (fut_px - self.prices) / self.prices
            
            # Map returns to transition timestamps
            df[f"ret_{h}d"] = df["timestamp"].map(ret_series)

        df["transition_pair"] = df["from_regime"] + " -> " + df["to_regime"]
        return df

    def analyze_alpha(self) -> pd.DataFrame:
        """Aggregate forward returns by transition pair and compute significance."""
        df = self.compute_forward_returns()
        if df.empty:
            return pd.DataFrame()

        results = []
        # Also compute unconditional returns for significance testing
        unconditional_returns = {}
        for h in self.horizons:
            fut_px = self.prices.shift(-h)
            unconditional_returns[h] = ((fut_px - self.prices) / self.prices).dropna().values

        for (rtype, pair), grp in df.groupby(["regime_type", "transition_pair"]):
            if len(grp) < 2:
                continue
                
            row = {
                "regime_type": rtype,
                "transition_pair": pair,
                "count": len(grp)
            }
            
            for h in self.horizons:
                rets = grp[f"ret_{h}d"].dropna().values
                if len(rets) < 2:
                    row[f"avg_ret_{h}d"] = np.nan
                    row[f"t_stat_{h}d"] = np.nan
                    row[f"p_val_{h}d"] = np.nan
                    continue

                avg_ret = float(np.mean(rets))
                row[f"avg_ret_{h}d"] = round(avg_ret, 6)

                # Statistical Significance: Two-sample t-test vs unconditional returns
                uncond = unconditional_returns[h]
                if len(uncond) > 2:
                    t_stat, p_val = stats.ttest_ind(rets, uncond, equal_var=False)
                    row[f"t_stat_{h}d"] = round(float(t_stat), 4)
                    row[f"p_val_{h}d"] = round(float(p_val), 4)
                else:
                    row[f"t_stat_{h}d"] = np.nan
                    row[f"p_val_{h}d"] = np.nan

            results.append(row)

        res_df = pd.DataFrame(results)
        if not res_df.empty:
            res_df = res_df.sort_values(["regime_type", "count"], ascending=[True, False]).reset_index(drop=True)
        return res_df

    def transition_frequency_matrix(self, regime_type: str = "volatility") -> pd.DataFrame:
        """Count matrix of transitions."""
        if self.transitions.empty:
            return pd.DataFrame()
        
        df = self.transitions[self.transitions["regime_type"] == regime_type]
        if df.empty:
            return pd.DataFrame()
            
        return pd.crosstab(df["from_regime"], df["to_regime"])
