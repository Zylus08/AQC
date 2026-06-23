"""
aqc/strategies/microprice_alpha/microprice_research.py
========================================================
Research report generator for the Microprice Alpha.

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

from aqc.alpha import AlphaMetrics
from aqc.strategies.microprice_alpha.microprice_alpha import MicropriceAlpha

logger = logging.getLogger(__name__)


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError as exc:
        raise ImportError("matplotlib required. pip install matplotlib") from exc


class MicropriceResearchReport:
    """Generate research reports for Microprice Alpha.

    Evaluates signal quality, PnL, and IC over time.

    Parameters
    ----------
    test_data:
        DataFrame for out-of-sample testing. Must contain snapshots and 'target_dir'.
    output_dir:
        Directory for saving plots.
    """

    def __init__(
        self,
        test_data: pd.DataFrame,
        output_dir: str = "reports/research/microprice",
    ) -> None:
        self.test_data = test_data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._alpha: Optional[MicropriceAlpha] = None
        self._predictions: Optional[pd.Series] = None

    def run_evaluation(self, **kwargs) -> pd.DataFrame:
        """Run the alpha evaluation on test data.

        Parameters
        ----------
        kwargs:
            Passed to the MicropriceAlpha constructor.

        Returns
        -------
        pd.DataFrame
            Performance metrics.
        """
        self._alpha = MicropriceAlpha(**kwargs)
        
        # We need the full history for z-score features
        features = self._alpha._feature_engine.extract(self.test_data)
        aligned = pd.concat([features, self.test_data["target_dir"]], axis=1).dropna()
        
        X = aligned.drop(columns=["target_dir"])
        y_true = aligned["target_dir"]

        self._predictions = self._alpha.predict(X)
        metrics = self._alpha.evaluate(self._predictions, y_true)

        return pd.DataFrame([metrics.to_dict()], index=[self._alpha.name])

    def plot_signal_distribution(self, save: bool = True):
        """Plot the distribution of microprice alpha scores."""
        plt = _require_matplotlib()
        plt.style.use("dark_background")

        if self._predictions is None:
            raise ValueError("Run run_evaluation() first.")

        fig, ax = plt.subplots(figsize=(10, 6), facecolor="#0d1117")
        ax.set_facecolor("#161b22")

        ax.hist(self._predictions, bins=50, color="#FFB74D", alpha=0.8, edgecolor="#21262d")
        
        ax.axvline(0, color="#888", linewidth=1, linestyle=":")
        ax.axvline(0.5, color="#81C784", linewidth=1.5, linestyle="--", label="Long Threshold")
        ax.axvline(-0.5, color="#F06292", linewidth=1.5, linestyle="--", label="Short Threshold")

        ax.set_title("Microprice Alpha Score Distribution", color="white")
        ax.set_xlabel("Score", color="white")
        ax.set_ylabel("Frequency", color="white")
        ax.legend(facecolor="#21262d", labelcolor="white")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

        plt.tight_layout()

        if save:
            fig.savefig(self.output_dir / "microprice_score_dist.png", facecolor=fig.get_facecolor())

        return fig
