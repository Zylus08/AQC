"""
aqc/strategies/liquidity_alpha/liquidity_research.py
======================================================
Research report generator for the Liquidity Alpha.

Evaluates how liquidity events (like depth drops) correlate
with future volatility and returns.

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import numpy as np

from aqc.strategies.liquidity_alpha.liquidity_alpha import LiquidityAlpha

logger = logging.getLogger(__name__)


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError as exc:
        raise ImportError("matplotlib required. pip install matplotlib") from exc


class LiquidityResearchReport:
    """Generate research reports for Liquidity Alpha."""

    def __init__(
        self,
        test_data: pd.DataFrame,
        output_dir: str = "reports/research/liquidity",
    ) -> None:
        self.test_data = test_data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._alpha = LiquidityAlpha()

    def run_evaluation(self) -> pd.DataFrame:
        """Run the alpha evaluation on test data."""
        features = self._alpha._feature_engine.extract(self.test_data)
        
        if "target_dir" in self.test_data.columns:
            aligned = pd.concat([features, self.test_data["target_dir"]], axis=1).dropna()
            X = aligned.drop(columns=["target_dir"])
            y_true = aligned["target_dir"]
        else:
            # Create synthetic target if missing for basic evaluation
            mid = self.test_data["mid_price"]
            future_ret = mid.shift(-1) / mid - 1
            y_true = np.sign(future_ret).fillna(0)
            X = features.loc[y_true.index]

        preds = self._alpha.predict(X)
        metrics = self._alpha.evaluate(preds, y_true)

        return pd.DataFrame([metrics.to_dict()], index=[self._alpha.name])

    def plot_liquidity_events(self, save: bool = True):
        """Plot liquidity regime events vs mid price."""
        plt = _require_matplotlib()
        plt.style.use("dark_background")

        features = self._alpha._feature_engine.extract(self.test_data)
        
        # Merge with price
        df = pd.concat([self.test_data["mid_price"], features], axis=1).dropna()

        fig, ax1 = plt.subplots(figsize=(12, 6), facecolor="#0d1117")
        ax1.set_facecolor("#161b22")

        ax1.plot(df.index, df["mid_price"], color="white", alpha=0.8, linewidth=1, label="Mid Price")
        
        # Highlight shock events
        shocks = df[df["is_liquidity_shock"] == 1]
        if not shocks.empty:
            ax1.scatter(shocks.index, shocks["mid_price"], color="#F06292", s=50, zorder=5, label="Liquidity Shock")

        ax1.set_title("Liquidity Shocks vs Price Action", color="white")
        ax1.set_ylabel("Price", color="white")
        ax1.tick_params(colors="white")
        
        ax2 = ax1.twinx()
        ax2.plot(df.index, df["depth_zscore"], color="#4FC3F7", alpha=0.4, linewidth=1, label="Depth Z-Score")
        ax2.axhline(-2.0, color="#FFB74D", linestyle="--", alpha=0.5, label="Shock Threshold")
        ax2.set_ylabel("Depth Z-Score", color="white")
        ax2.tick_params(colors="white")

        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left", facecolor="#21262d", labelcolor="white")

        for ax in [ax1, ax2]:
            for spine in ax.spines.values():
                spine.set_edgecolor("#30363d")

        plt.tight_layout()

        if save:
            fig.savefig(self.output_dir / "liquidity_events.png", facecolor=fig.get_facecolor())

        return fig
