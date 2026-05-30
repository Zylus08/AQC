"""
aqc/diagnostics/diagnostics_dashboard.py
==========================================
Single-page HTML diagnostics dashboard that embeds all analysis panels.

Author: Saksham Mishra — AlgoQuant Club
"""
from __future__ import annotations
import logging, base64, io
from pathlib import Path
from typing import Optional
import pandas as pd

logger = logging.getLogger(__name__)

# Dark-mode CSS
_CSS = """
:root { --bg: #0d1117; --card: #161b22; --border: #30363d; --txt: #c9d1d9;
        --accent: #4FC3F7; --warn: #FF7043; --ok: #66BB6A; --bad: #EF5350; }
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Inter', 'Segoe UI', sans-serif; background: var(--bg); color: var(--txt); padding: 24px; }
h1 { color: var(--accent); text-align: center; margin-bottom: 32px; font-size: 1.8rem; }
h2 { color: #e6edf3; margin: 18px 0 10px; font-size: 1.1rem; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(520px, 1fr)); gap: 18px; }
.card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 20px; }
.card img { width: 100%; border-radius: 8px; margin-top: 8px; }
table { border-collapse: collapse; width: 100%; margin-top: 8px; font-size: 0.85rem; }
th, td { border: 1px solid var(--border); padding: 6px 10px; text-align: right; }
th { background: #21262d; color: var(--accent); }
.score-bar { height: 18px; border-radius: 4px; }
.score-ok { background: var(--ok); } .score-warn { background: var(--warn); } .score-bad { background: var(--bad); }
.score-label { display: inline-block; min-width: 36px; text-align: center; font-weight: 600; }
.violations { list-style: none; padding: 0; } .violations li { color: var(--warn); padding: 3px 0; }
.violations li::before { content: "!  "; color: var(--bad); font-weight: 700; }
footer { text-align: center; color: #484f58; margin-top: 32px; font-size: 0.8rem; }
"""


def _embed_img(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    data = p.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f'<img src="data:image/png;base64,{b64}" alt="{p.stem}" />'


class PortfolioDiagnosticsDashboard:
    """Generate a single-page HTML diagnostics dashboard.

    Parameters
    ----------
    results : dict        from DiagnosticsEngine.run_all()
    plots_dir : str       where .png files were saved
    output_path : str     where to write the HTML file
    """
    def __init__(self, results: dict, plots_dir: str = "reports",
                 output_path: str = "dashboard/diagnostics_dashboard.html") -> None:
        self.results = results
        self.plots_dir = plots_dir
        self.output_path = output_path

    def generate(self) -> str:
        html = [
            "<!DOCTYPE html><html><head><meta charset='utf-8'>",
            "<title>AQC Portfolio Diagnostics</title>",
            f"<style>{_CSS}</style></head><body>",
            "<h1>AQC Portfolio Diagnostics Dashboard</h1>",
        ]

        # Validation Score Card
        val = self.results.get("validation", {})
        if val:
            html.append('<div class="card"><h2>Validation Scores</h2><table>')
            html.append("<tr><th>Category</th><th>Score</th><th>Bar</th></tr>")
            for k, v in val.items():
                cls = "score-ok" if v >= 80 else ("score-warn" if v >= 60 else "score-bad")
                bar = f'<div class="score-bar {cls}" style="width:{v}%"></div>'
                html.append(f"<tr><td style='text-align:left'>{k}</td>"
                           f"<td><span class='score-label'>{v}</span></td><td>{bar}</td></tr>")
            html.append("</table>")
            viols = self.results.get("violations", [])
            if viols:
                html.append("<h2>Violations</h2><ul class='violations'>")
                for v in viols:
                    html.append(f"<li>{v}</li>")
                html.append("</ul>")
            html.append("</div>")

        # Plot panels
        html.append('<div class="grid">')
        panels = [
            ("Leverage Analysis", "leverage_over_time.png"),
            ("Exposure Analysis", "exposure_over_time.png"),
            ("Risk Budget", "risk_budget_utilisation.png"),
            ("Position Analysis", "position_size_distribution.png"),
            ("Regime Performance", "regime_performance_heatmap.png"),
            ("Forecast Validation", "forecast_error_distribution.png"),
            ("Attribution", "attribution_breakdown.png"),
            ("Drawdown Forensics", "drawdown_forensics.png"),
        ]
        for title, filename in panels:
            img_path = f"{self.plots_dir}/{filename}"
            img_html = _embed_img(img_path)
            if img_html:
                html.append(f'<div class="card"><h2>{title}</h2>{img_html}</div>')

        html.append("</div>")

        # Summary tables
        html.append('<div class="grid">')
        # Leverage stats
        lev = self.results.get("leverage")
        if lev:
            html.append('<div class="card"><h2>Leverage Stats</h2><table>')
            html.append("<tr><th>Metric</th><th>Value</th></tr>")
            for k, v in lev.__dict__.items():
                html.append(f"<tr><td style='text-align:left'>{k}</td><td>{v}</td></tr>")
            html.append("</table></div>")

        # Attribution
        attr = self.results.get("attribution", {})
        if attr:
            html.append('<div class="card"><h2>Attribution</h2><table>')
            html.append("<tr><th>Source</th><th>Contribution</th></tr>")
            for k, v in attr.items():
                html.append(f"<tr><td style='text-align:left'>{k}</td><td>{v*100:+.2f}%</td></tr>")
            html.append("</table></div>")
        html.append("</div>")

        html.append("<footer>Generated by AQC Diagnostics Framework</footer>")
        html.append("</body></html>")

        output = "\n".join(html)
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.output_path).write_text(output, encoding="utf-8")
        logger.info("Dashboard saved to %s", self.output_path)
        return output
