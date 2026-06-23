from typing import Dict, Any, List, Tuple

class MicropriceEstimator:
    """Implements Microprice and Weighted MidPrice."""
    
    def calculate_microprice(self, best_bid: float, best_bid_vol: float, 
                             best_ask: float, best_ask_vol: float) -> float:
        """
        Calculates volume-weighted midprice (Microprice).
        """
        if best_bid_vol + best_ask_vol == 0:
            return (best_bid + best_ask) / 2
            
        # Microprice = (Bid * AskVol + Ask * BidVol) / (BidVol + AskVol)
        # Note: This balances the price towards the side with MORE volume
        return (best_bid * best_ask_vol + best_ask * best_bid_vol) / (best_bid_vol + best_ask_vol)
        
    def calculate_fair_value(self, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]], levels: int = 5) -> float:
        """Calculates fair value incorporating multiple levels of depth."""
        bid_px_vol = sum(p * v for p, v in bids[:levels])
        ask_px_vol = sum(p * v for p, v in asks[:levels])
        
        bid_vol = sum(v for _, v in bids[:levels])
        ask_vol = sum(v for _, v in asks[:levels])
        
        if bid_vol + ask_vol == 0:
            return 0.0
            
        return (bid_px_vol + ask_px_vol) / (bid_vol + ask_vol)
