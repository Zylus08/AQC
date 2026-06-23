from typing import Dict, Any, List, Tuple

class ImbalanceEngine:
    """Computes order book imbalance metrics."""
    
    def compute_imbalance(self, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]], levels: int = 5) -> float:
        """
        Computes (BidVol - AskVol) / (BidVol + AskVol) for top N levels.
        """
        bid_vol = sum(v for _, v in bids[:levels])
        ask_vol = sum(v for _, v in asks[:levels])
        
        if bid_vol + ask_vol == 0:
            return 0.0
            
        return (bid_vol - ask_vol) / (bid_vol + ask_vol)
        
    def get_imbalance_metrics(self, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]]) -> Dict[str, float]:
        return {
            "imbalance_top_1": self.compute_imbalance(bids, asks, 1),
            "imbalance_top_5": self.compute_imbalance(bids, asks, 5),
            "imbalance_top_10": self.compute_imbalance(bids, asks, 10),
            "imbalance_full": self.compute_imbalance(bids, asks, len(bids))
        }
