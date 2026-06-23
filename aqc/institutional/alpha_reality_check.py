from typing import Dict, Any

class AlphaRealityCheck:
    """Answers if alpha survives real trading costs."""
    
    def check_survival(self, expected_alpha: float, 
                       spread_cost: float, 
                       commission_cost: float, 
                       slippage_cost: float, 
                       market_impact_cost: float, 
                       adverse_selection_cost: float) -> Dict[str, Any]:
        """
        Subtracts execution and microstructure costs from theoretical alpha.
        All inputs should be in consistent units (e.g., bps or absolute returns).
        """
        total_costs = spread_cost + commission_cost + slippage_cost + market_impact_cost + adverse_selection_cost
        net_alpha = expected_alpha - total_costs
        
        survives = net_alpha > 0
        
        return {
            "expected_alpha": expected_alpha,
            "total_costs": total_costs,
            "net_alpha": net_alpha,
            "survives": survives,
            "breakdown": {
                "spread_cost": spread_cost,
                "commission": commission_cost,
                "slippage": slippage_cost,
                "market_impact": market_impact_cost,
                "adverse_selection": adverse_selection_cost
            }
        }
