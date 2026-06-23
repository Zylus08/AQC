import pandas as pd
from typing import Dict, Any, List
from .paper_statistics import PaperStatistics
from .paper_health import PaperHealth

class PaperMonitor:
    """Tracks live paper trading metrics over rolling windows."""
    
    def __init__(self):
        self.stats = PaperStatistics()
        self.health = PaperHealth()
        self.daily_returns = []
        
    def add_daily_return(self, ret: float):
        self.daily_returns.append(ret)
        
    def get_rolling_metrics(self) -> Dict[str, Any]:
        """Generate 5d, 20d, 40d, 60d metrics"""
        windows = [5, 20, 40, 60]
        results = {}
        
        for w in windows:
            if len(self.daily_returns) >= w:
                recent = self.daily_returns[-w:]
                res = self.stats.calculate_metrics(recent, [])
                results[f"{w}d"] = res
            else:
                results[f"{w}d"] = {"cagr": 0.0, "sharpe": 0.0, "avg_slippage": 0.0}
                
        return results
