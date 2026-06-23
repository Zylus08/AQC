from typing import Dict, Any, List

class ExecutionValidator:
    """Validates if execution quality matches expectations (slippage, fill rate, costs)."""
    
    def __init__(self, thresholds: Dict[str, float] = None):
        self.thresholds = thresholds or {
            "slippage_increase": 0.50, # 50% worse slippage than expected
            "fill_rate_drop": 0.10,    # 10% drop in fill rate
            "cost_increase": 0.25      # 25% increase in execution costs
        }
        
    def analyze(self, expected_slippage: float, observed_slippage: float,
                expected_fill_rate: float, observed_fill_rate: float,
                expected_cost: float, observed_cost: float) -> Dict[str, Any]:
        """
        Analyzes execution metrics.
        """
        slippage_change = (observed_slippage - expected_slippage) / expected_slippage if expected_slippage > 0 else 0
        fill_rate_change = expected_fill_rate - observed_fill_rate # raw drop
        cost_change = (observed_cost - expected_cost) / expected_cost if expected_cost > 0 else 0
        
        # Calculate score
        score = 100.0
        
        if slippage_change > 0:
            score -= (slippage_change / self.thresholds['slippage_increase']) * 20
            
        if fill_rate_change > 0:
            score -= (fill_rate_change / self.thresholds['fill_rate_drop']) * 20
            
        if cost_change > 0:
            score -= (cost_change / self.thresholds['cost_increase']) * 20
            
        score = max(0.0, min(100.0, score))
        status = self._get_status(score)
        
        return {
            "slippage_change": slippage_change,
            "fill_rate_change": fill_rate_change,
            "cost_change": cost_change,
            "execution_score": score,
            "status": status,
            "warnings": self._generate_warnings(slippage_change, fill_rate_change, cost_change)
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
            
    def _generate_warnings(self, slip: float, fill: float, cost: float) -> List[str]:
        warnings = []
        if slip > self.thresholds['slippage_increase']:
            warnings.append(f"Slippage worsened by {slip:.1%} (threshold: {self.thresholds['slippage_increase']:.1%})")
        if fill > self.thresholds['fill_rate_drop']:
            warnings.append(f"Fill rate dropped by {fill:.1%} (threshold: {self.thresholds['fill_rate_drop']:.1%})")
        if cost > self.thresholds['cost_increase']:
            warnings.append(f"Execution costs increased by {cost:.1%} (threshold: {self.thresholds['cost_increase']:.1%})")
        return warnings
