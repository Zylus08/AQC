from typing import Dict, Any

class DeploymentReadinessEngine:
    """Computes a 0-100 Deployment Readiness Score."""
    
    def __init__(self):
        self.weights = {
            "forward_validation": 0.25,
            "execution_quality": 0.25,
            "alpha_confidence": 0.25,
            "capacity": 0.15,
            "paper_trading": 0.10
        }
        
    def calculate_score(self, fw_val_score: float, exec_quality_score: float, 
                        alpha_prob_alive: float, capacity_score: float, 
                        paper_trading_score: float) -> Dict[str, Any]:
        """
        Calculates final deployment readiness score.
        All inputs should be 0-100 (except alpha_prob_alive which is 0.0-1.0 and will be scaled to 100).
        """
        
        alpha_score = alpha_prob_alive * 100
        
        final_score = (
            fw_val_score * self.weights["forward_validation"] +
            exec_quality_score * self.weights["execution_quality"] +
            alpha_score * self.weights["alpha_confidence"] +
            capacity_score * self.weights["capacity"] +
            paper_trading_score * self.weights["paper_trading"]
        )
        
        status = self._get_status(final_score)
        
        return {
            "deployment_score": final_score,
            "status": status,
            "components": {
                "forward_validation": fw_val_score,
                "execution_quality": exec_quality_score,
                "alpha_confidence": alpha_score,
                "capacity": capacity_score,
                "paper_trading": paper_trading_score
            }
        }
        
    def _get_status(self, score: float) -> str:
        if score >= 85:
            return "READY"
        elif score >= 70:
            return "MONITOR"
        elif score >= 50:
            return "LIMITED CAPITAL"
        else:
            return "NOT DEPLOYABLE"
