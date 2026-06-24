"""
aqc/research/hypothesis_lab/experiment_runner.py
==================================================
Runs baseline tests for a hypothesis.

Links a Hypothesis IDEA to an actual AlphaBase implementation.

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging

import pandas as pd

from aqc.alpha.alpha_factory import AlphaFactory
from aqc.research.hypothesis_lab.hypothesis import AlphaHypothesis, HypothesisStatus
from aqc.research.hypothesis_lab.hypothesis_registry import HypothesisRegistry

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Executes baseline evaluations for hypotheses."""

    def __init__(self, registry: HypothesisRegistry) -> None:
        self.registry = registry
        self.factory = AlphaFactory()

    def run_baseline(
        self,
        hypothesis_id: str,
        alpha_class_name: str,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
    ) -> bool:
        """Run a baseline evaluation for a hypothesis.

        Parameters
        ----------
        hypothesis_id:
            ID of the hypothesis to test.
        alpha_class_name:
            Name of the registered AlphaBase to instantiate.
        train_data:
            Data for fitting.
        test_data:
            Data for out-of-sample evaluation.

        Returns
        -------
        bool
            True if the baseline ran successfully.
        """
        hypothesis = self.registry.get(hypothesis_id)
        if not hypothesis:
            logger.error("Hypothesis %s not found.", hypothesis_id)
            return False

        hypothesis.update_status(HypothesisStatus.TESTING, "Starting baseline experiment.")
        self.registry.save()

        try:
            # Instantiate
            alpha = self.factory.create(alpha_class_name, train_data=train_data)
            
            # Predict and Evaluate
            # Simplified: assuming test_data has target_dir and features directly
            # In a full system, we extract features first.
            if "target_dir" not in test_data.columns:
                raise ValueError("test_data must contain 'target_dir'")

            # Handle alphas with feature engines
            if hasattr(alpha, "_feature_engine"):
                try:
                    import inspect
                    sig = inspect.signature(alpha._feature_engine.extract)
                    if len(sig.parameters) >= 2 and "trades" in sig.parameters:
                        features = alpha._feature_engine.extract(test_data, test_data)
                    else:
                        features = alpha._feature_engine.extract(test_data)
                except Exception as e:
                    logger.warning("Feature extraction failed: %s", e)
                    features = test_data
            else:
                features = test_data

            aligned = pd.concat([features, test_data["target_dir"]], axis=1).dropna()
            X = aligned.drop(columns=["target_dir"])
            y = aligned["target_dir"]

            preds = alpha.predict(X)
            metrics = alpha.evaluate(preds, y)

            # Store results
            hypothesis.test_results = metrics.to_dict()
            
            # Determine success heuristics
            if metrics.sharpe_ratio > 0.5 and metrics.information_coefficient > 0.01:
                hypothesis.update_status(HypothesisStatus.PROMISING, f"Baseline passed. Sharpe: {metrics.sharpe_ratio:.2f}")
            else:
                hypothesis.update_status(HypothesisStatus.FAILED, f"Baseline failed. Sharpe: {metrics.sharpe_ratio:.2f}")

            self.registry.save()
            return True

        except Exception as e:
            logger.error("Experiment failed for %s: %s", hypothesis_id, e)
            hypothesis.update_status(HypothesisStatus.FAILED, f"Exception during baseline: {e}")
            self.registry.save()
            return False
