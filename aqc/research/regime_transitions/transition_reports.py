"""
aqc/research/regime_transitions/transition_reports.py
=======================================================
Generate text and CSV reports for regime transition alpha research.

Author: Saksham Mishra — AlgoQuant Club
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class TransitionReportGenerator:
    """Generate reports for transition alpha analysis.

    Parameters
    ----------
    alpha_df : pd.DataFrame
        Output of TransitionAlphaAnalyzer.analyze_alpha().
    horizons : list[int]
        Forward horizons used in analysis.
    """

    def __init__(self, alpha_df: pd.DataFrame, horizons: Optional[list[int]] = None) -> None:
        self.alpha_df = alpha_df
        self.horizons = horizons or [1, 3, 5, 10, 20]

    def build_report(self) -> str:
        """Build text report."""
        if self.alpha_df.empty:
            return "No transition data available."

        sep = "=" * 80
        thin = "-" * 80
        lines = ["", sep, "  AQC REGIME TRANSITION ALPHA REPORT", sep, ""]

        for rtype, grp in self.alpha_df.groupby("regime_type"):
            lines += [f"  {rtype.upper()} TRANSITIONS", thin]
            
            # Header
            header = f"  {'Transition Pair':25s} | {'Count':5s} |"
            for h in self.horizons:
                header += f" {h}d Ret   Sig |"
            lines.append(header)
            lines.append(thin)

            for _, row in grp.iterrows():
                pair = row["transition_pair"]
                count = row["count"]
                
                line = f"  {pair:25s} | {count:5d} |"
                
                for h in self.horizons:
                    ret = row.get(f"avg_ret_{h}d", float("nan"))
                    pval = row.get(f"p_val_{h}d", 1.0)
                    
                    if pd.isna(ret):
                        ret_str = "  N/A  "
                        sig_str = "   "
                    else:
                        ret_str = f"{ret*100:>+6.2f}%"
                        # Significance stars: *** < 0.01, ** < 0.05, * < 0.10
                        if pval < 0.01: sig_str = "***"
                        elif pval < 0.05: sig_str = "** "
                        elif pval < 0.10: sig_str = "*  "
                        else: sig_str = "   "
                    
                    line += f" {ret_str} {sig_str} |"
                
                lines.append(line)
            
            lines.append("")

        lines += ["  Significance: *** p<0.01, ** p<0.05, * p<0.10 (vs unconditional returns)"]
        lines += [sep, ""]
        return "\n".join(lines)

    def print_report(self) -> None:
        print(self.build_report())

    def save_csv(self, output_dir: str = "reports") -> None:
        p = Path(output_dir)
        p.mkdir(parents=True, exist_ok=True)
        if not self.alpha_df.empty:
            self.alpha_df.to_csv(p / "transition_alpha_report.csv", index=False)
            
            # Extract just significance
            sig_cols = ["regime_type", "transition_pair", "count"] + [f"p_val_{h}d" for h in self.horizons]
            sig_df = self.alpha_df[[c for c in sig_cols if c in self.alpha_df.columns]]
            sig_df.to_csv(p / "transition_significance.csv", index=False)
            
            logger.info("Transition alpha reports saved to %s", output_dir)
