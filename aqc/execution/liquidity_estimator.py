from typing import Dict, Any

class LiquidityEstimator:
    """Estimates current market liquidity based on order book and volume profile."""
    
    def estimate_liquidity(self, spread: float, book_depth: float, recent_volume: float, adv: float) -> str:
        """Returns the liquidity regime."""
        if adv <= 0:
            return "UNKNOWN"
            
        rel_vol = recent_volume / (adv / 390) # Assuming 390 trading mins
        
        if spread > 0.05 and book_depth < 10000:
            return "CRISIS"
        elif spread > 0.02 or rel_vol < 0.5:
            return "THIN"
        elif rel_vol > 2.0:
            return "STRESS" # High volume, possibly wide spread
        else:
            return "NORMAL"
