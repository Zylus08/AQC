import math
from typing import Dict, Any

class MarketImpactModel:
    """Models market impact including temporary and permanent components."""
    
    def __init__(self, temp_coeff: float = 0.1, perm_coeff: float = 0.05):
        self.temp_coeff = temp_coeff
        self.perm_coeff = perm_coeff
        
    def estimate_impact(self, order_qty: float, adv: float, vol: float, 
                        is_aggressive: bool = True) -> Dict[str, float]:
        """
        Estimates temporary and permanent impact in basis points.
        Uses Square-Root model for temporary impact, Linear model for permanent.
        """
        if adv <= 0:
            return {"temporary_bps": 0.0, "permanent_bps": 0.0, "total_bps": 0.0}
            
        participation = abs(order_qty) / adv
        
        # Temporary impact (Square-root)
        temp_impact = self.temp_coeff * vol * math.sqrt(participation) * 10000.0
        
        # Permanent impact (Linear)
        perm_impact = self.perm_coeff * vol * participation * 10000.0
        
        if not is_aggressive:
            temp_impact *= 0.5 # Passive orders incur less temporary impact
            
        return {
            "temporary_bps": float(temp_impact),
            "permanent_bps": float(perm_impact),
            "total_bps": float(temp_impact + perm_impact)
        }
