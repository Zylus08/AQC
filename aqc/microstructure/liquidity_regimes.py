from typing import Dict, Any

class LiquidityRegimes:
    """Detects microstructure liquidity regimes."""
    
    def detect_regime(self, spread: float, depth: float, impact: float) -> str:
        if spread > 0.10 and impact > 0.05:
            return "CRISIS"
        elif spread > 0.05 or depth < 1000:
            return "THIN"
        elif impact > 0.02:
            return "STRESS"
        else:
            return "NORMAL"
