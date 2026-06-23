from typing import Dict, Any, List

class OrderflowAnalyzer:
    """Analyzes historical trades for order flow characteristics."""
    
    def analyze_flow(self, trades: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        trades should have 'price', 'volume', 'direction' (or inferred from lee-ready).
        """
        buy_vol = sum(t['volume'] for t in trades if t.get('direction') == 'BUY')
        sell_vol = sum(t['volume'] for t in trades if t.get('direction') == 'SELL')
        
        total_vol = buy_vol + sell_vol
        
        ofi = (buy_vol - sell_vol) / total_vol if total_vol > 0 else 0.0
        
        return {
            "buy_volume": float(buy_vol),
            "sell_volume": float(sell_vol),
            "total_volume": float(total_vol),
            "order_flow_imbalance": float(ofi)
        }
