"""
aqc/diagnostics/attribution.py
=================================
Performance attribution: where did returns actually come from?

Decomposes total return into:
  1. Alpha contribution
  2. Leverage contribution
  3. Volatility targeting contribution
  4. Regime filter contribution
  5. Residual

Author: Saksham Mishra — AlgoQuant Club
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Optional
import numpy as np, pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class AttributionResult:
    total_return: float = 0.0
    alpha_contribution: float = 0.0
    leverage_contribution: float = 0.0
    vol_target_contribution: float = 0.0
    regime_filter_contribution: float = 0.0
    residual: float = 0.0

    def to_dict(self) -> dict:
        return {
            "Alpha": self.alpha_contribution,
            "Leverage": self.leverage_contribution,
            "Vol Targeting": self.vol_target_contribution,
            "Regime Filter": self.regime_filter_contribution,
            "Residual": self.residual,
            "Total": self.total_return,
        }

    def pct_dict(self) -> dict:
        """Contributions as percentage of total."""
        total = abs(self.total_return) if abs(self.total_return) > 1e-10 else 1.0
        return {k: round(v / total * 100, 2) for k, v in self.to_dict().items()}


class PerformanceAttributionEngine:
    """Decompose returns by source.

    Approach (Brinson-style simplified):
        Total Return = Alpha + Leverage Effect + Vol Target Effect + Regime Effect + Residual

    Where:
        - Alpha ≈ return of equal-weight unlevered portfolio
        - Leverage ≈ (leveraged return - unleveraged return)
        - Vol Target ≈ (vol-targeted return - baseline return)
        - Regime Filter ≈ (regime-filtered return - unfiltered return)

    Parameters
    ----------
    baseline_returns  : pd.Series   returns without enhancements
    vol_target_returns: pd.Series   returns with vol targeting only
    regime_returns    : pd.Series   returns with regime filter + vol targeting
    combined_returns  : pd.Series   returns with everything
    leverage_series   : pd.Series   gross leverage per bar
    """
    def __init__(
        self,
        baseline_returns: pd.Series,
        vol_target_returns: Optional[pd.Series] = None,
        regime_returns: Optional[pd.Series] = None,
        combined_returns: Optional[pd.Series] = None,
        leverage_series: Optional[pd.Series] = None,
    ) -> None:
        self.baseline = baseline_returns
        self.vol_target = vol_target_returns
        self.regime = regime_returns
        self.combined = combined_returns
        self.leverage = leverage_series

    def compute(self) -> AttributionResult:
        total_ret = float((1 + self.baseline).prod() - 1) if self.baseline is not None else 0.0

        # Alpha = baseline return (the unenhanced strategy)
        alpha = total_ret

        # Leverage contribution
        lev_contrib = 0.0
        if self.leverage is not None and len(self.leverage) > 0:
            avg_lev = float(self.leverage.mean())
            if avg_lev > 1.001:
                # Return above 1x leverage is leverage contribution
                unlev_return = total_ret / avg_lev if avg_lev > 0 else total_ret
                lev_contrib = total_ret - unlev_return
                alpha = unlev_return

        # Vol targeting contribution
        vt_contrib = 0.0
        if self.vol_target is not None and len(self.vol_target) > 0:
            vt_ret = float((1 + self.vol_target).prod() - 1)
            vt_contrib = vt_ret - total_ret

        # Regime filter contribution
        reg_contrib = 0.0
        if self.regime is not None and self.vol_target is not None:
            vt_ret = float((1 + self.vol_target).prod() - 1)
            reg_ret = float((1 + self.regime).prod() - 1)
            reg_contrib = reg_ret - vt_ret

        # Combined total
        combined_total = total_ret
        if self.combined is not None and len(self.combined) > 0:
            combined_total = float((1 + self.combined).prod() - 1)

        residual = combined_total - (alpha + lev_contrib + vt_contrib + reg_contrib)

        return AttributionResult(
            total_return=round(combined_total, 6),
            alpha_contribution=round(alpha, 6),
            leverage_contribution=round(lev_contrib, 6),
            vol_target_contribution=round(vt_contrib, 6),
            regime_filter_contribution=round(reg_contrib, 6),
            residual=round(residual, 6),
        )

    def save_report(self, path: str = "reports/attribution_report.csv") -> None:
        from pathlib import Path
        result = self.compute()
        df = pd.DataFrame([result.to_dict()]).T
        df.columns = ["contribution"]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path)

    def plot(self, save: bool = True, output_dir: str = "reports") -> None:
        import matplotlib.pyplot as plt
        from pathlib import Path
        result = self.compute()
        d = result.to_dict()
        # Remove Total for the waterfall
        labels = [k for k in d if k != "Total"]
        values = [d[k] * 100 for k in labels]
        cumulative = np.cumsum(values)

        fig, ax = plt.subplots(figsize=(12, 6), facecolor="#0d1117")
        ax.set_facecolor("#161b22")
        colors = ["#4FC3F7", "#FF7043", "#66BB6A", "#AB47BC", "#FFB74D"]
        bottoms = [0] + list(cumulative[:-1])
        for i, (lbl, val, bot) in enumerate(zip(labels, values, bottoms)):
            c = colors[i % len(colors)]
            ax.bar(lbl, val, bottom=bot, color=c, alpha=0.85, edgecolor="#21262d")
            ax.text(i, bot + val / 2, f"{val:+.1f}%", ha="center", va="center",
                    color="white", fontsize=10, fontweight="bold")

        # Total bar
        ax.bar("Total", cumulative[-1], color="#FFD54F", alpha=0.9, edgecolor="#21262d")
        ax.text(len(labels), cumulative[-1] / 2, f"{cumulative[-1]:.1f}%",
                ha="center", va="center", color="black", fontsize=10, fontweight="bold")

        ax.axhline(0, color="white", lw=0.8)
        ax.set_title("Performance Attribution Waterfall", color="white", fontsize=14, fontweight="bold")
        ax.set_ylabel("Contribution (%)", color="white")
        for s in ax.spines.values(): s.set_edgecolor("#30363d")
        ax.tick_params(colors="white")
        plt.tight_layout()
        if save:
            p = Path(output_dir) / "attribution_breakdown.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
