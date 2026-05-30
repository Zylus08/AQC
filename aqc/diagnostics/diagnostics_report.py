"""
aqc/diagnostics/diagnostics_report.py
=======================================
Console + CSV report generation for diagnostics results.

Author: Saksham Mishra — AlgoQuant Club
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional
import pandas as pd

logger = logging.getLogger(__name__)


class DiagnosticsReportGenerator:
    """Generate formatted diagnostic reports from engine results.

    Parameters
    ----------
    results : dict   output of DiagnosticsEngine.run_all()
    """
    def __init__(self, results: dict) -> None:
        self.results = results

    def print_report(self) -> None:
        report = self.build_report()
        print(report)

    def build_report(self) -> str:
        sep = "=" * 70
        thin = "-" * 70
        lines = [
            "", sep,
            "  AQC PORTFOLIO DIAGNOSTICS REPORT",
            sep, "",
        ]

        # Validation Score
        val = self.results.get("validation", {})
        if val:
            lines += ["  VALIDATION SCORES", thin]
            for k, v in val.items():
                bar = "#" * (v // 5) + " " * (20 - v // 5)
                lines.append(f"  {k:20s} : [{bar}] {v}/100")
            lines.append("")

        # Violations
        violations = self.results.get("violations", [])
        if violations:
            lines += ["  VIOLATIONS DETECTED", thin]
            for v in violations:
                lines.append(f"  [!] {v}")
            lines.append("")

        # Leverage
        lev = self.results.get("leverage")
        if lev:
            lines += ["  LEVERAGE ANALYSIS", thin]
            lines.append(f"  Avg Gross Leverage  : {lev.avg_gross:.4f}")
            lines.append(f"  Max Gross Leverage  : {lev.max_gross:.4f}")
            lines.append(f"  Avg Net Leverage    : {lev.avg_net:.4f}")
            lines.append(f"  % Leveraged         : {lev.pct_leveraged*100:.1f}%")
            lines.append(f"  Utilisation         : {lev.leverage_utilisation*100:.1f}%")
            lines.append("")

        # Exposure
        exp = self.results.get("exposure")
        if exp:
            lines += ["  EXPOSURE ANALYSIS", thin]
            lines.append(f"  Avg Gross Exposure  : {exp.avg_gross:.4f}")
            lines.append(f"  Avg Long / Short    : {exp.avg_long:.4f} / {exp.avg_short:.4f}")
            lines.append(f"  % Fully Invested    : {exp.pct_fully_invested*100:.1f}%")
            lines.append("")

        # Risk Budget
        rb = self.results.get("risk_budget")
        if rb:
            lines += ["  RISK BUDGET ANALYSIS", thin]
            lines.append(f"  Avg Utilisation      : {rb.avg_utilisation:.4f}")
            lines.append(f"  Max Utilisation      : {rb.max_utilisation:.4f}")
            lines.append(f"  % Over Budget        : {rb.pct_over_budget*100:.1f}%")
            lines.append(f"  % Under 50%          : {rb.pct_under_50*100:.1f}%")
            lines.append("")

        # Position
        pos = self.results.get("position")
        if pos:
            lines += ["  POSITION ANALYSIS", thin]
            lines.append(f"  Avg Position Size   : ${pos.avg_size:,.2f}")
            lines.append(f"  Max Position Size   : ${pos.max_size:,.2f}")
            lines.append(f"  Max Weight          : {pos.max_weight:.2%}")
            lines.append(f"  HHI Concentration   : {pos.hhi_concentration:.4f}")
            lines.append(f"  Avg Turnover        : {pos.avg_turnover:.4f}")
            lines.append("")

        # Attribution
        attr = self.results.get("attribution", {})
        if attr:
            lines += ["  PERFORMANCE ATTRIBUTION", thin]
            for k, v in attr.items():
                lines.append(f"  {k:20s} : {v*100:+.2f}%")
            lines.append("")

        lines += [sep, ""]
        return "\n".join(lines)

    def save_summary(self, path: str = "reports/diagnostics_summary.csv") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        flat = {}
        for section, data in self.results.items():
            if hasattr(data, '__dict__'):
                for k, v in data.__dict__.items():
                    flat[f"{section}.{k}"] = v
            elif isinstance(data, dict):
                for k, v in data.items():
                    flat[f"{section}.{k}"] = v
        pd.DataFrame([flat]).T.to_csv(path, header=["value"])
