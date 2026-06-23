from typing import Dict, Any

class MicrostructureFeatures:
    """Aggregates all microstructure features for predictive modeling."""
    
    def extract_features(self, ofi: float, vpin: float, vr: float, adverse_selection: float) -> Dict[str, float]:
        return {
            "ofi": float(ofi),
            "vpin": float(vpin),
            "variance_ratio": float(vr),
            "adverse_selection": float(adverse_selection)
        }
