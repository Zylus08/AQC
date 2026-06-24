"""
aqc/research/hypothesis_lab/hypothesis_tracker.py
===================================================
Tracks hypotheses through their lifecycle and generates summaries.

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging

import pandas as pd

from aqc.research.hypothesis_lab.hypothesis_registry import HypothesisRegistry

logger = logging.getLogger(__name__)


class HypothesisTracker:
    """Provides reporting and lifecycle tracking for hypotheses."""

    def __init__(self, registry: HypothesisRegistry) -> None:
        self.registry = registry

    def get_summary_dataframe(self) -> pd.DataFrame:
        """Get a DataFrame summarizing all hypotheses."""
        data = []
        for h in self.registry.get_all():
            sharpe = h.test_results.get("sharpe_ratio", 0.0)
            ic = h.test_results.get("information_coefficient", 0.0)
            
            data.append({
                "ID": h.id,
                "Title": h.title,
                "Status": h.status.value,
                "Creator": h.creator,
                "Features": len(h.feature_set),
                "Baseline_Sharpe": sharpe,
                "Baseline_IC": ic,
                "Created": h.created_at,
                "Updated": h.updated_at,
            })

        if not data:
            return pd.DataFrame()
            
        df = pd.DataFrame(data)
        return df.sort_values("Created", ascending=False)

    def print_funnel_stats(self) -> None:
        """Log the funnel statistics (IDEA -> TESTED -> DEPLOYED)."""
        df = self.get_summary_dataframe()
        if df.empty:
            logger.info("Hypothesis funnel is empty.")
            return

        counts = df["Status"].value_counts()
        total = len(df)
        
        logger.info("=== HYPOTHESIS FUNNEL ===")
        logger.info("Total Ideas: %d", total)
        for status, count in counts.items():
            pct = (count / total) * 100
            logger.info("%s: %d (%.1f%%)", status, count, pct)
