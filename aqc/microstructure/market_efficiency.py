import numpy as np
from typing import List, Dict, Any

class MarketEfficiency:
    """Measures Variance Ratio and Autocorrelation."""
    
    def calculate_autocorrelation(self, returns: List[float], lag: int = 1) -> float:
        if len(returns) <= lag:
            return 0.0
        
        y1 = returns[:-lag]
        y2 = returns[lag:]
        
        if np.std(y1) == 0 or np.std(y2) == 0:
            return 0.0
            
        return float(np.corrcoef(y1, y2)[0, 1])
        
    def calculate_variance_ratio(self, prices: List[float], q: int = 5) -> float:
        """
        Variance ratio test (VR = Var(q-period return) / (q * Var(1-period return)))
        """
        if len(prices) <= q:
            return 1.0
            
        prices_arr = np.array(prices)
        r1 = np.diff(np.log(prices_arr))
        
        rq = np.log(prices_arr[q:]) - np.log(prices_arr[:-q])
        
        var_1 = np.var(r1)
        var_q = np.var(rq)
        
        if var_1 == 0:
            return 1.0
            
        return float(var_q / (q * var_1))
