from typing import Dict, Any
from .market_impact_model import MarketImpactModel
from .slippage_model import SlippageModel

class ExecutionCostEngine:
    """Combines all execution models to estimate total execution cost."""
    
    def __init__(self):
        self.impact_model = MarketImpactModel()
        self.slippage_model = SlippageModel(fixed_bps=1.0)
        
    def estimate_total_cost(self, price: float, order_qty: float, adv: float, 
                            vol: float, spread: float, is_aggressive: bool = True) -> Dict[str, Any]:
        
        impact = self.impact_model.estimate_impact(order_qty, adv, vol, is_aggressive)
        
        # Absolute slippage
        slip_abs = self.slippage_model.estimate_slippage(price, spread, vol, order_qty, adv)
        slip_bps = (slip_abs / price) * 10000.0 if price > 0 else 0.0
        
        total_bps = impact["total_bps"] + slip_bps
        
        return {
            "total_cost_bps": float(total_bps),
            "impact_bps": impact["total_bps"],
            "slippage_bps": float(slip_bps),
            "details": impact
        }
