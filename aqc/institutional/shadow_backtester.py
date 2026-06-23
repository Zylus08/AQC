import pandas as pd
import numpy as np
from typing import Dict, Any, List

class ShadowBacktester:
    """Compares expected trades from research with actual live trades."""
    
    def __init__(self):
        self.divergences = []
        
    def compare_trades(self, expected_trades: List[Dict], actual_trades: List[Dict]) -> Dict[str, Any]:
        """
        expected_trades and actual_trades should be lists of dicts with:
        'timestamp', 'symbol', 'direction', 'price', 'quantity'
        """
        exp_df = pd.DataFrame(expected_trades)
        act_df = pd.DataFrame(actual_trades)
        
        if exp_df.empty and act_df.empty:
            return {"status": "MATCH", "missed_trades": 0, "unexpected_trades": 0, "avg_price_divergence": 0.0, "total_expected": 0, "total_actual": 0}
            
        missed = 0
        unexpected = 0
        price_diffs = []
        
        if exp_df.empty and not act_df.empty:
            unexpected = len(act_df)
        elif not exp_df.empty and act_df.empty:
            missed = len(exp_df)
        else:
            # Match trades by symbol and direction
            for _, exp_trade in exp_df.iterrows():
                matches = act_df[(act_df['symbol'] == exp_trade['symbol']) & 
                                 (act_df['direction'] == exp_trade['direction'])]
                if matches.empty:
                    missed += 1
                else:
                    # Find closest in price for simplicity in this implementation
                    closest_idx = (matches['price'] - exp_trade['price']).abs().idxmin()
                    closest = matches.loc[closest_idx]
                    if exp_trade['price'] > 0:
                        price_diffs.append(abs(closest['price'] - exp_trade['price']) / exp_trade['price'])
                    
            unexpected = max(0, len(act_df) - (len(exp_df) - missed))
            
        avg_price_diff = float(np.mean(price_diffs)) if price_diffs else 0.0
        
        return {
            "missed_trades": missed,
            "unexpected_trades": unexpected,
            "avg_price_divergence": avg_price_diff,
            "total_expected": len(exp_df),
            "total_actual": len(act_df)
        }
