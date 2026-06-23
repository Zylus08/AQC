"""
aqc/strategies/orderflow_alpha/orderflow_research.py
======================================================
Research report generator for the Order Flow Alpha.

Evaluates how VPIN and flow imbalance correlate with future returns.

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import numpy as np

from aqc.strategies.orderflow_alpha.orderflow_alpha import OrderFlowAlpha

logger = logging.getLogger(__name__)


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError as exc:
        raise ImportError("matplotlib required. pip install matplotlib") from exc


class OrderFlowResearchReport:
    """Generate research reports for Order Flow Alpha."""

    def __init__(
        self,
        trades: pd.DataFrame,
        snapshots: pd.DataFrame,
        output_dir: str = "reports/research/orderflow",
    ) -> None:
        self.trades = trades
        self.snapshots = snapshots
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._alpha = OrderFlowAlpha()

    def run_evaluation(self) -> pd.DataFrame:
        """Run the alpha evaluation."""
        features = self._alpha._feature_engine.extract(self.trades, self.snapshots)
        
        if "target_dir" in self.snapshots.columns:
            aligned = pd.concat([features, self.snapshots["target_dir"]], axis=1).dropna()
            X = aligned.drop(columns=["target_dir"])
            y_true = aligned["target_dir"]
        else:
            mid = self.snapshots["mid_price"]
            future_ret = mid.shift(-1) / mid - 1
            y_true = np.sign(future_ret).fillna(0)
            X = features.loc[y_true.index]

        preds = self._alpha.predict(X)
        metrics = self._alpha.evaluate(preds, y_true)

        return pd.DataFrame([metrics.to_dict()], index=[self._alpha.name])

    def plot_vpin_and_flow(self, save: bool = True):
        """Plot VPIN and smoothed flow imbalance."""
        plt = _require_matplotlib()
        plt.style.use("dark_background")

        features = self._alpha._feature_engine.extract(self.trades, self.snapshots)
        
        fig, ax1 = plt.subplots(figsize=(12, 6), facecolor="#0d1117")
        ax1.set_facecolor("#161b22")

        ax1.plot(features.index, features["vpin"], color="#F06292", alpha=0.8, linewidth=1.5, label="VPIN (Toxicity)")
        ax1.axhline(self._alpha.vpin_threshold, color="#CE93D8", linestyle="--", alpha=0.5, label="VPIN Threshold")
        ax1.set_title("VPIN and Order Flow Imbalance", color="white")
        ax1.set_ylabel("VPIN", color="white")
        ax1.tick_params(colors="white")
        
        ax2 = ax1.twinx()
        ax2.fill_between(
            features.index, 
            0, 
            features["flow_imbalance_smooth"], 
            where=(features["flow_imbalance_smooth"] > 0), 
            color="#81C784", alpha=0.4, label="Buy Flow"
        )
        ax2.fill_between(
            features.index, 
            0, 
            features["flow_imbalance_smooth"], 
            where=(features["flow_imbalance_smooth"] < 0), 
            color="#E53935", alpha=0.4, label="Sell Flow"
        )
        ax2.set_ylabel("Smoothed Flow Imbalance", color="white")
        ax2.tick_params(colors="white")

        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left", facecolor="#21262d", labelcolor="white")

        for ax in [ax1, ax2]:
            for spine in ax.spines.values():
                spine.set_edgecolor("#30363d")

        plt.tight_layout()

        if save:
            fig.savefig(self.output_dir / "orderflow_vpin.png", facecolor=fig.get_facecolor())

        return fig
