from typing import Dict, Any, List

class EarlyWarningSystem:
    """Collects warnings from analyzers and categorizes alerts."""
    
    def __init__(self):
        self.alerts = []
        
    def add_warnings(self, category: str, warnings: List[str]):
        for warning in warnings:
            # Simple heuristic for criticality
            level = "CRITICAL" if "collapse" in warning.lower() or "> 50%" in warning or "significantly" in warning else "WARNING"
            self.alerts.append({
                "category": category,
                "level": level,
                "message": warning
            })
            
    def get_alerts(self):
        return self.alerts
    
    def clear_alerts(self):
        self.alerts = []


class StrategyHealthEngine:
    """Combines all validation scores to generate an overall health report."""
    
    def __init__(self):
        # Weights for different components
        self.weights = {
            "alpha": 0.35,
            "signal": 0.20,
            "forecast": 0.15,
            "regime": 0.15,
            "execution": 0.15
        }
        
    def generate_report(self, alpha_res: Dict, signal_res: Dict, 
                        forecast_res: Dict, regime_res: Dict, execution_res: Dict) -> Dict[str, Any]:
        """
        Combines results from all analyzers to generate a unified strategy health report.
        """
        overall_score = (
            alpha_res.get('decay_score', 100) * self.weights['alpha'] +
            signal_res.get('signal_score', 100) * self.weights['signal'] +
            forecast_res.get('forecast_score', 100) * self.weights['forecast'] +
            regime_res.get('regime_score', 100) * self.weights['regime'] +
            execution_res.get('execution_score', 100) * self.weights['execution']
        )
        
        status = self._get_status(overall_score)
        
        # Aggregate warnings
        ews = EarlyWarningSystem()
        ews.add_warnings("Alpha Decay", alpha_res.get("warnings", []))
        ews.add_warnings("Signal Stability", signal_res.get("warnings", []))
        ews.add_warnings("Forecast Stability", forecast_res.get("warnings", []))
        ews.add_warnings("Regime Drift", regime_res.get("warnings", []))
        ews.add_warnings("Execution Quality", execution_res.get("warnings", []))
        
        return {
            "overall_score": overall_score,
            "status": status,
            "component_scores": {
                "alpha_score": alpha_res.get('decay_score', 100),
                "signal_score": signal_res.get('signal_score', 100),
                "forecast_score": forecast_res.get('forecast_score', 100),
                "regime_score": regime_res.get('regime_score', 100),
                "execution_score": execution_res.get('execution_score', 100)
            },
            "alerts": ews.get_alerts()
        }
        
    def _get_status(self, score: float) -> str:
        if score >= 90:
            return "HEALTHY"
        elif score >= 70:
            return "MONITOR"
        elif score >= 50:
            return "WARNING"
        else:
            return "CRITICAL"
