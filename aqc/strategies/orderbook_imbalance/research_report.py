"""
aqc/strategies/orderbook_imbalance/research_report.py
=======================================================
Research report generator for the Order Book Imbalance Alpha.

Evaluates multiple models, generates feature importance, and plots PnL.

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import roc_auc_score, precision_score, recall_score
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

from aqc.alpha import AlphaMetrics
from aqc.strategies.orderbook_imbalance.imbalance_alpha import OrderBookImbalanceAlpha

logger = logging.getLogger(__name__)


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError as exc:
        raise ImportError("matplotlib required. pip install matplotlib") from exc


class ImbalanceResearchReport:
    """Generate research reports for Order Book Imbalance Alpha.

    Trains multiple models, evaluates their performance out-of-sample,
    and produces comparison metrics and plots.

    Parameters
    ----------
    train_data:
        DataFrame for training. Must contain snapshots and 'target_dir'.
    test_data:
        DataFrame for out-of-sample testing. Same format as train.
    output_dir:
        Directory for saving plots.
    """

    def __init__(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        output_dir: str = "reports/research/orderbook",
    ) -> None:
        if not ML_AVAILABLE:
            raise ImportError("scikit-learn required for research reports.")

        self.train_data = train_data
        self.test_data = test_data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._results: dict[str, dict] = {}
        self._alphas: dict[str, OrderBookImbalanceAlpha] = {}

    def run_suite(
        self,
        models: tuple[str, ...] = ("logistic", "rf", "xgboost", "lightgbm"),
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Run the full research suite across multiple models.

        Parameters
        ----------
        models:
            List of model types to evaluate.
        kwargs:
            Passed to the alpha constructors.

        Returns
        -------
        pd.DataFrame
            Comparison metrics across all tested models.
        """
        for model_type in models:
            logger.info("Evaluating %s...", model_type)
            alpha = OrderBookImbalanceAlpha(
                name=f"ob_imb_{model_type}",
                model_type=model_type,
                **kwargs,
            )
            alpha.fit(self.train_data)

            # Test features
            X_test = alpha._feature_engine.extract(self.test_data)
            aligned = pd.concat([X_test, self.test_data["target_dir"]], axis=1).dropna()
            X = aligned.drop(columns=["target_dir"])
            y_true = aligned["target_dir"]

            preds = alpha.predict(X)

            # Standard alpha metrics
            metrics = alpha.evaluate(preds, y_true)

            # ML specific metrics (assuming binary classification mapped to 0, 1 internally)
            # For simplicity, we convert predictions back to discrete classes based on threshold
            y_pred_class = np.zeros_like(preds)
            y_pred_class[preds > alpha.threshold] = 1
            y_pred_class[preds < -alpha.threshold] = -1

            # Ignore 0s for precision/recall calculation to focus on actioned signals
            mask = y_pred_class != 0
            if mask.sum() > 0:
                prec = precision_score(y_true[mask], y_pred_class[mask], average="macro", zero_division=0)
                rec = recall_score(y_true[mask], y_pred_class[mask], average="macro", zero_division=0)
            else:
                prec, rec = 0.0, 0.0

            # ROC-AUC (requires probability-like scores mapped to 0-1)
            # This is an approximation since targets are -1, 0, 1.
            # Convert to binary: Up (1) vs Non-Up (-1, 0) for AUC calculation
            y_true_bin = (y_true == 1).astype(int)
            y_score_bin = (preds + 1) / 2.0  # map [-1, 1] to [0, 1]
            try:
                auc = roc_auc_score(y_true_bin, y_score_bin)
            except ValueError:
                auc = 0.5  # If only one class present

            res = metrics.to_dict()
            res.update({
                "roc_auc": round(float(auc), 4),
                "precision": round(float(prec), 4),
                "recall": round(float(rec), 4),
            })

            self._results[model_type] = res
            self._alphas[model_type] = alpha

        return pd.DataFrame(self._results).T

    def plot_feature_importance(self, model_type: str = "xgboost", save: bool = True):
        """Plot feature importances for a specific model."""
        plt = _require_matplotlib()
        plt.style.use("dark_background")

        if model_type not in self._alphas:
            raise ValueError(f"Model {model_type} not evaluated yet.")

        alpha = self._alphas[model_type]
        imp = alpha._model.feature_importance()

        if imp.empty:
            logger.warning("No feature importances available for %s", model_type)
            return None

        fig, ax = plt.subplots(figsize=(10, 8), facecolor="#0d1117")
        ax.set_facecolor("#161b22")

        imp.head(15).sort_values().plot(kind="barh", ax=ax, color="#4FC3F7")

        ax.set_title(f"Top 15 Feature Importances ({model_type})", color="white")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

        plt.tight_layout()

        if save:
            fig.savefig(self.output_dir / f"feat_imp_{model_type}.png", facecolor=fig.get_facecolor())

        return fig
