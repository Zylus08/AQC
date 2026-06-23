from typing import List, Dict, Any

class AdverseSelection:
    """Calculates expected move after fill (markouts)."""
    
    def calculate_markout(self, execution_price: float, side: str, future_prices: List[float]) -> List[float]:
        """
        Calculates post-trade price movement relative to execution price.
        """
        sign = 1 if side == 'BUY' else -1
        
        markouts = []
        for p in future_prices:
            # Profit/loss from the perspective of the resting order
            markouts.append(sign * (p - execution_price) / execution_price)
            
        return markouts
        
    def estimate_adverse_selection_cost(self, markouts: List[float]) -> float:
        """
        Adverse selection is typically the negative markout at a specific horizon (e.g. 1min, 5min).
        """
        # Assume markouts[0] is 1-min, markouts[-1] is 5-min
        if not markouts:
            return 0.0
        # If markout is negative, it means price moved against the fill
        return float(-min(0, markouts[-1]))
