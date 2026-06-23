from typing import Dict, Any, List
import numpy as np

class ImbalancePredictor:
    """Predicts next tick direction based on order book features."""
    
    def __init__(self, model_type="logistic"):
        self.model_type = model_type
        self.is_trained = False
        self.model = None
        
        try:
            if model_type == "xgboost":
                from xgboost import XGBClassifier
                self.model = XGBClassifier()
            elif model_type == "lightgbm":
                from lightgbm import LGBMClassifier
                self.model = LGBMClassifier()
            elif model_type == "random_forest":
                from sklearn.ensemble import RandomForestClassifier
                self.model = RandomForestClassifier()
            else:
                from sklearn.linear_model import LogisticRegression
                self.model = LogisticRegression()
        except ImportError:
            # Fallback to a simple heuristic if libraries aren't available
            self.model = None
            
    def train(self, X: np.ndarray, y: np.ndarray):
        if self.model is not None:
            self.model.fit(X, y)
            self.is_trained = True
            
    def predict(self, features: np.ndarray) -> Dict[str, float]:
        """
        Predicts next tick direction probability.
        Features could be: [imbalance_top_1, imbalance_top_5, spread, microprice_deviation]
        """
        if not self.is_trained or self.model is None:
            # Simple heuristic based on first feature (assume it's imbalance)
            imbalance = features[0][0] if len(features.shape) > 1 else features[0]
            prob_up = 0.5 + 0.5 * imbalance # mapping [-1, 1] to [0, 1]
            return {"prob_up": float(prob_up), "prob_down": float(1 - prob_up)}
            
        probs = self.model.predict_proba(features)[0]
        # Assume class 1 is UP
        return {
            "prob_up": float(probs[1]) if len(probs) > 1 else 0.5,
            "prob_down": float(probs[0]) if len(probs) > 1 else 0.5
        }
