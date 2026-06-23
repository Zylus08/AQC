from typing import Dict, Any
from .execution_cost_engine import ExecutionCostEngine
from .execution_optimizer import ExecutionOptimizer

class ExecutionSimulator:
    """Simulates an order execution to predict slippage, fill rate, and costs."""
    
    def __init__(self):
        self.cost_engine = ExecutionCostEngine()
        self.optimizer = ExecutionOptimizer()
        
    def simulate_order(self, order_qty: float, price: float, adv: float, vol: float, spread: float) -> Dict[str, Any]:
        opt = self.optimizer.optimize_execution(order_qty, adv, "NORMAL")
        is_aggressive = opt["best_strategy"] == "AGGRESSIVE"
        
        costs = self.cost_engine.estimate_total_cost(price, order_qty, adv, vol, spread, is_aggressive)
        
        return {
            "simulated_strategy": opt["best_strategy"],
            "simulated_cost_bps": costs["total_cost_bps"],
            "estimated_fill_rate": 1.0 if is_aggressive else 0.95
        }
