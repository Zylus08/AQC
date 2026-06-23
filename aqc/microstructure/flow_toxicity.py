import numpy as np
from typing import List, Dict, Any

class FlowToxicity:
    """Calculates Volume Synchronized Probability of Informed Trading (VPIN)."""
    
    def calculate_vpin(self, volume_buckets: List[Dict[str, float]], volume_per_bucket: float) -> float:
        """
        volume_buckets: list of dicts with 'buy_vol' and 'sell_vol'
        """
        if not volume_buckets or volume_per_bucket <= 0:
            return 0.0
            
        total_imbalance = sum(abs(b.get('buy_vol', 0) - b.get('sell_vol', 0)) for b in volume_buckets)
        total_volume = len(volume_buckets) * volume_per_bucket
        
        return float(total_imbalance / total_volume) if total_volume > 0 else 0.0
