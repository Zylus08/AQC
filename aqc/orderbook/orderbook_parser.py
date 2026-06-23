import pandas as pd
from typing import Dict, Any, List, Tuple

class OrderbookParser:
    """Parses raw L2 order book data into structured format."""
    
    def parse_snapshot(self, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]]) -> Dict[str, Any]:
        """
        Parses a snapshot of the book.
        bids, asks are list of tuples: (price, volume)
        """
        # Sort bids descending, asks ascending
        bids_sorted = sorted(bids, key=lambda x: x[0], reverse=True)
        asks_sorted = sorted(asks, key=lambda x: x[0])
        
        return {
            "bids": bids_sorted,
            "asks": asks_sorted,
            "best_bid": bids_sorted[0][0] if bids_sorted else 0.0,
            "best_ask": asks_sorted[0][0] if asks_sorted else 0.0,
            "best_bid_vol": bids_sorted[0][1] if bids_sorted else 0.0,
            "best_ask_vol": asks_sorted[0][1] if asks_sorted else 0.0,
            "mid_price": (bids_sorted[0][0] + asks_sorted[0][0]) / 2 if bids_sorted and asks_sorted else 0.0
        }
