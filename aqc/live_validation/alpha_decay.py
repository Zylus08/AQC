from typing import Dict, Any, List
from .expectation_tracker import ExpectedPerformanceProfile

class AlphaDecayAnalyzer:
    """Analyzes alpha decay by comparing live metrics against expected metrics."""
    
    def __init__(self, thresholds: Dict[str, float] = None):
        """
        Args:
            thresholds: Dictionary of decay thresholds (e.g., {'sharpe_decay': 0.30})
        """
        # Default degradation thresholds (percentages e.g. 0.3 = 30% drop)
        self.thresholds = thresholds or {
            "sharpe_decay": 0.30,
            "return_decay": 0.30,
            "profit_factor_decay": 0.20,
            "win_rate_decay": 0.15
        }

    def analyze(self, expected: ExpectedPerformanceProfile, observed: Dict[str, float]) -> Dict[str, Any]:
        """
        Calculates decay across key metrics and generates a decay score.
        
        Args:
            expected: ExpectedPerformanceProfile object
            observed: Dictionary containing current live metrics ('sharpe', 'cagr', etc.)
        """
        # Calculate decays
        sharpe_decay = self._calc_decay(expected.sharpe, observed.get('sharpe', expected.sharpe))
        return_decay = self._calc_decay(expected.cagr, observed.get('cagr', expected.cagr))
        profit_factor_decay = self._calc_decay(expected.profit_factor, observed.get('profit_factor', expected.profit_factor))
        win_rate_decay = self._calc_decay(expected.win_rate, observed.get('win_rate', expected.win_rate))
        
        # Calculate overall decay score (0 to 100, where 100 is no decay, 0 is max decay)
        # We penalize more heavily for large decays
        avg_decay = (sharpe_decay + return_decay + profit_factor_decay + win_rate_decay) / 4.0
        decay_score = max(0, 100 - (avg_decay * 100))
        
        status = self._get_status(decay_score)
        
        return {
            "sharpe_decay": sharpe_decay,
            "return_decay": return_decay,
            "profit_factor_decay": profit_factor_decay,
            "win_rate_decay": win_rate_decay,
            "decay_score": decay_score,
            "status": status,
            "warnings": self._generate_warnings(sharpe_decay, return_decay, profit_factor_decay, win_rate_decay)
        }
        
    def _calc_decay(self, expected: float, observed: float) -> float:
        """Calculates percentage decay from expected to observed."""
        if expected <= 0 and observed <= 0:
            return 0.0
        if expected <= 0:
            return 0.0 if observed >= expected else 1.0 # Edge case
        
        decay = (expected - observed) / expected
        return max(0.0, decay) # Only consider negative drift as decay

    def _get_status(self, score: float) -> str:
        if score >= 90:
            return "HEALTHY"
        elif score >= 70:
            return "MONITOR"
        elif score >= 50:
            return "WARNING"
        else:
            return "CRITICAL"
            
    def _generate_warnings(self, sharpe: float, ret: float, pf: float, win: float) -> List[str]:
        warnings = []
        if sharpe > self.thresholds['sharpe_decay']:
            warnings.append(f"Sharpe decay {sharpe:.1%} exceeds threshold {self.thresholds['sharpe_decay']:.1%}")
        if ret > self.thresholds['return_decay']:
            warnings.append(f"Return decay {ret:.1%} exceeds threshold {self.thresholds['return_decay']:.1%}")
        if pf > self.thresholds['profit_factor_decay']:
            warnings.append(f"Profit factor decay {pf:.1%} exceeds threshold {self.thresholds['profit_factor_decay']:.1%}")
        if win > self.thresholds['win_rate_decay']:
            warnings.append(f"Win rate decay {win:.1%} exceeds threshold {self.thresholds['win_rate_decay']:.1%}")
        return warnings
