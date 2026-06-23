from typing import Dict, Any, List
from datetime import datetime
from dataclasses import dataclass, field

@dataclass
class ModelVersion:
    version_id: str
    strategy_name: str
    deployment_date: datetime
    expected_profile_id: str
    is_active: bool
    health_history: List[Dict[str, Any]] = field(default_factory=list)

class RetrainingAdvisor:
    """Generates recommendations on whether a model should be retrained."""
    
    def generate_recommendation(self, current_health: Dict[str, Any], 
                                expected_sharpe: float, current_sharpe: float) -> Dict[str, Any]:
        """
        Determines retraining recommendation based on health score and key metrics.
        """
        score = current_health.get('overall_score', 100)
        
        if score < 50 or (expected_sharpe > 0 and current_sharpe / expected_sharpe < 0.5):
            action = "RETRAIN MODEL"
            confidence = min(100.0, 100.0 - score + 20.0) # Heuristic confidence
        elif score < 70:
            action = "MONITOR CLOSELY"
            confidence = 80.0
        else:
            action = "HOLD"
            confidence = 95.0
            
        return {
            "recommendation": action,
            "confidence": confidence,
            "current_sharpe": current_sharpe,
            "expected_sharpe": expected_sharpe,
            "reason": f"Overall Health Score is {score:.1f}"
        }


class ModelGovernanceEngine:
    """Tracks model versions, deployment dates, and health history."""
    
    def __init__(self):
        self.models: Dict[str, ModelVersion] = {}
        self.advisor = RetrainingAdvisor()
        
    def register_model(self, version_id: str, strategy_name: str, expected_profile_id: str) -> None:
        self.models[version_id] = ModelVersion(
            version_id=version_id,
            strategy_name=strategy_name,
            deployment_date=datetime.now(),
            expected_profile_id=expected_profile_id,
            is_active=True
        )
        
    def update_health(self, version_id: str, health_report: Dict[str, Any]) -> None:
        if version_id in self.models:
            report_with_ts = health_report.copy()
            report_with_ts['timestamp'] = datetime.now().isoformat()
            self.models[version_id].health_history.append(report_with_ts)
            
    def get_model_history(self, version_id: str) -> List[Dict[str, Any]]:
        if version_id in self.models:
            return self.models[version_id].health_history
        return []
        
    def generate_registry_report(self) -> List[Dict[str, Any]]:
        registry = []
        for v_id, model in self.models.items():
            latest_health = model.health_history[-1]['overall_score'] if model.health_history else None
            registry.append({
                "version_id": model.version_id,
                "strategy_name": model.strategy_name,
                "deployment_date": model.deployment_date.isoformat(),
                "is_active": model.is_active,
                "latest_health_score": latest_health
            })
        return registry
        
    def advise_retraining(self, version_id: str, current_sharpe: float, expected_sharpe: float) -> Dict[str, Any]:
        if version_id not in self.models or not self.models[version_id].health_history:
            return {"recommendation": "INSUFFICIENT DATA", "confidence": 0.0, "reason": "No health history available."}
            
        latest_health = self.models[version_id].health_history[-1]
        return self.advisor.generate_recommendation(latest_health, expected_sharpe, current_sharpe)
