from typing import Dict, Any

class PaperHealth:
    """Assesses survival probability after 20-60 day unseen data periods."""
    
    def assess_survival(self, days_active: int, current_sharpe: float, expected_sharpe: float) -> Dict[str, Any]:
        """
        Did the strategy survive X days of unseen market data?
        """
        if days_active < 20:
            status = "INITIALIZING"
            survival_prob = 0.5 # Neutral
        else:
            # Simple heuristic: closer to expected_sharpe -> higher survival prob
            ratio = current_sharpe / max(expected_sharpe, 0.01)
            survival_prob = min(1.0, max(0.0, ratio))
            
            if days_active >= 60 and survival_prob > 0.7:
                status = "SURVIVED"
            elif survival_prob > 0.5:
                status = "ON TRACK"
            else:
                status = "FAILING"
                
        return {
            "days_active": days_active,
            "survival_probability": survival_prob,
            "status": status,
            "is_ready": (status == "SURVIVED")
        }
