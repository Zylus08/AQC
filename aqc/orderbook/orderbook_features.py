from typing import Dict, Any, List, Tuple

class OrderbookFeatures:
    """Computes aggregate features from orderbook."""
    
    def compute_features(self, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]], 
                         recent_trades: List[Dict] = None) -> Dict[str, float]:
        
        best_bid = bids[0][0] if bids else 0.0
        best_ask = asks[0][0] if asks else 0.0
        spread = best_ask - best_bid
        
        depth_bids = sum(v for _, v in bids)
        depth_asks = sum(v for _, v in asks)
        
        # Book pressure (weighted depth)
        bid_pressure = sum(v / ((best_bid - p) + 0.01) for p, v in bids)
        ask_pressure = sum(v / ((p - best_ask) + 0.01) for p, v in asks)
        
        trade_intensity = len(recent_trades) if recent_trades else 0
        
        return {
            "spread": float(spread),
            "depth_bids": float(depth_bids),
            "depth_asks": float(depth_asks),
            "total_depth": float(depth_bids + depth_asks),
            "bid_pressure": float(bid_pressure),
            "ask_pressure": float(ask_pressure),
            "trade_intensity": float(trade_intensity)
        }
