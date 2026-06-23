from typing import Dict, Any

class ExecutionOptimizer:
    """Optimizes execution by comparing aggressive vs passive, TWAP, VWAP, POV."""
    
    def __init__(self):
        pass
        
    def optimize_execution(self, order_qty: float, adv: float, urgency: str = "NORMAL") -> Dict[str, Any]:
        """
        Determines the optimal execution method.
        urgency: "HIGH", "NORMAL", "LOW"
        """
        participation = order_qty / adv if adv > 0 else 1.0
        
        # Simple heuristic rule engine
        if urgency == "HIGH":
            strategy = "AGGRESSIVE"
            target_participation = min(0.20, participation * 2)
        elif urgency == "LOW" and participation > 0.05:
            strategy = "TWAP"
            target_participation = 0.05
        elif participation > 0.10:
            strategy = "VWAP"
            target_participation = 0.10
        elif participation > 0.02:
            strategy = "POV"
            target_participation = 0.05
        else:
            strategy = "PASSIVE"
            target_participation = participation
            
        return {
            "best_strategy": strategy,
            "target_participation_rate": float(target_participation),
            "estimated_slices": max(1, int(participation / target_participation) if target_participation > 0 else 1)
        }
