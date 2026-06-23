import numpy as np
import pandas as pd
from typing import Dict, Any, List

class PaperStatistics:
    """Calculates daily returns, slippage, and execution stats."""
    
    def calculate_metrics(self, daily_returns: List[float], slippage_history: List[float]) -> Dict[str, Any]:
        if not daily_returns:
            return {"cagr": 0.0, "sharpe": 0.0, "avg_slippage": 0.0}
            
        returns_arr = np.array(daily_returns)
        cagr = (1 + np.mean(returns_arr)) ** 252 - 1
        
        std = np.std(returns_arr)
        if std > 0:
            sharpe = np.mean(returns_arr) / std * np.sqrt(252)
        else:
            sharpe = 0.0
        
        avg_slippage = np.mean(slippage_history) if slippage_history else 0.0
        
        return {
            "cagr": float(cagr),
            "sharpe": float(sharpe),
            "avg_slippage": float(avg_slippage)
        }
