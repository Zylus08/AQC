import pandas as pd
import numpy as np
from typing import Dict, Any, List

class LiveResearchTracker:
    """Tracks and compares equity curves between Research, Paper, and Live."""
    
    def __init__(self):
        self.research_equity = []
        self.paper_equity = []
        self.live_equity = []
        
    def add_data_point(self, timestamp: str, research_val: float, paper_val: float, live_val: float):
        self.research_equity.append({"timestamp": timestamp, "value": research_val})
        self.paper_equity.append({"timestamp": timestamp, "value": paper_val})
        self.live_equity.append({"timestamp": timestamp, "value": live_val})
        
    def compute_metrics(self) -> Dict[str, Any]:
        """Computes tracking error and performance drift."""
        if len(self.research_equity) < 2 or len(self.live_equity) < 2:
            return {"tracking_error": 0.0, "performance_drift": 0.0, "cum_research": 0.0, "cum_live": 0.0}
            
        research_df = pd.DataFrame(self.research_equity).set_index("timestamp")
        live_df = pd.DataFrame(self.live_equity).set_index("timestamp")
        
        # Combine
        df = research_df.join(live_df, lsuffix='_research', rsuffix='_live', how='inner')
        if df.empty or len(df) < 2:
            return {"tracking_error": 0.0, "performance_drift": 0.0, "cum_research": 0.0, "cum_live": 0.0}
            
        # Calculate returns
        df['ret_research'] = df['value_research'].pct_change()
        df['ret_live'] = df['value_live'].pct_change()
        df = df.dropna()
        
        if df.empty:
            return {"tracking_error": 0.0, "performance_drift": 0.0, "cum_research": 0.0, "cum_live": 0.0}
            
        # Tracking Error: Std dev of difference in returns (annualized)
        diff = df['ret_live'] - df['ret_research']
        tracking_error = diff.std() * np.sqrt(252) # Assuming daily returns
        
        # Performance Drift: Total cumulative difference
        cum_research = (1 + df['ret_research']).prod() - 1
        cum_live = (1 + df['ret_live']).prod() - 1
        performance_drift = cum_live - cum_research
        
        return {
            "tracking_error": float(tracking_error),
            "performance_drift": float(performance_drift),
            "cum_research": float(cum_research),
            "cum_live": float(cum_live)
        }
