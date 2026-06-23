"""
aqc/strategies/orderbook_imbalance/prediction_models.py
=========================================================
ML prediction suite for order book alphas.

Provides a unified scikit-learn compatible interface for Logistic Regression,
Random Forest, XGBoost, and LightGBM.

Author: Saksham Mishra — AlgoQuant Club
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    import xgboost as xgb
    import lightgbm as lgb
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logger.warning(
        "ML libraries missing. Run: "
        "pip install scikit-learn xgboost lightgbm"
    )

logger = logging.getLogger(__name__)


class ImbalancePredictionSuite:
    """Unified ML prediction suite for alpha scores.

    Wraps multiple ML models into a standard interface for training
    and inference.  Outputs are mapped to ``[-1.0, 1.0]`` alpha scores
    (where 1.0 is highest conviction long, -1.0 highest conviction short).

    Parameters
    ----------
    model_type:
        One of ``"logistic"``, ``"rf"``, ``"xgboost"``, ``"lightgbm"``.
    **kwargs:
        Passed directly to the underlying model constructor.

    Examples
    --------
    >>> suite = ImbalancePredictionSuite("xgboost", max_depth=4)
    >>> suite.fit(X_train, y_train)
    >>> scores = suite.predict(X_test)
    """

    def __init__(self, model_type: str = "logistic", **kwargs: Any) -> None:
        if not ML_AVAILABLE:
            raise ImportError(
                "PredictionSuite requires scikit-learn, xgboost, and lightgbm."
            )

        self.model_type = model_type.lower()
        self.kwargs = kwargs
        self.model = self._build_model()
        self._feature_names: Optional[list[str]] = None
        self._is_fitted = False

    def _build_model(self) -> Any:
        if self.model_type == "logistic":
            return LogisticRegression(max_iter=1000, **self.kwargs)
        elif self.model_type == "rf":
            return RandomForestClassifier(n_estimators=100, **self.kwargs)
        elif self.model_type == "xgboost":
            return xgb.XGBClassifier(
                use_label_encoder=False,
                eval_metric="logloss",
                **self.kwargs,
            )
        elif self.model_type == "lightgbm":
            return lgb.LGBMClassifier(**self.kwargs)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the model.

        Parameters
        ----------
        X:
            Feature matrix.
        y:
            Target labels (usually -1, 0, 1).  Will be mapped to 0, 1, 2
            if required by the underlying model.
        """
        self._feature_names = list(X.columns)

        # Map y from {-1, 0, 1} to {0, 1, 2} for XGBoost/LightGBM compatibility
        y_mapped = y.copy()
        if set(y.unique()).issubset({-1, 0, 1}):
            y_mapped = y_mapped + 1  # [-1,0,1] -> [0,1,2]

        logger.info("Training %s on %d samples...", self.model_type, len(X))
        self.model.fit(X.values, y_mapped.values)
        self._is_fitted = True
        logger.info("Training complete.")

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict alpha scores in [-1.0, 1.0].

        The score is derived from class probabilities.
        If classes are [Down, Flat, Up], score = P(Up) - P(Down).

        Parameters
        ----------
        X:
            Feature matrix.

        Returns
        -------
        pd.Series
            Alpha scores in [-1.0, 1.0].
        """
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        # Ensure feature alignment
        if self._feature_names is not None:
            missing = [f for f in self._feature_names if f not in X.columns]
            if missing:
                raise ValueError(f"Missing features in prediction data: {missing}")
            X_align = X[self._feature_names]
        else:
            X_align = X

        probs = self.model.predict_proba(X_align.values)

        # Map probabilities to a [-1, 1] score
        # Assuming classes are [0: Down, 1: Flat, 2: Up] (mapped from -1, 0, 1)
        if probs.shape[1] == 3:
            p_down = probs[:, 0]
            p_up = probs[:, 2]
            scores = p_up - p_down
        elif probs.shape[1] == 2:
            # Binary classification (Down, Up) mapped to 0, 1
            # Which corresponds to -1, 1
            # P(Up) is prob of class 1.
            p_up = probs[:, 1]
            p_down = probs[:, 0]
            scores = p_up - p_down
        else:
            # Fallback
            preds = self.model.predict(X_align.values)
            # Map back from [0, 1, 2] -> [-1, 0, 1]
            scores = preds - 1

        return pd.Series(scores, index=X.index, name="alpha_score")

    def feature_importance(self) -> pd.Series:
        """Return feature importances if available.

        Returns
        -------
        pd.Series
            Importance scores indexed by feature name.
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted.")

        importances = None
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            importances = np.abs(self.model.coef_[0])

        if importances is None:
            logger.warning("Model %s does not expose feature importances.", self.model_type)
            return pd.Series(dtype=float)

        s = pd.Series(importances, index=self._feature_names)
        return s.sort_values(ascending=False)
