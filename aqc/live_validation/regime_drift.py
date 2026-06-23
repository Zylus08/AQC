import numpy as np
from typing import Dict, Any, List
from scipy.stats import chisquare

class RegimeDriftAnalyzer:
    """Detects regime drift by comparing live market regimes against training regimes."""
    
    def __init__(self, thresholds: Dict[str, float] = None):
        self.thresholds = thresholds or {
            "psi_threshold": 0.25,
            "p_value_threshold": 0.05
        }
        
    def analyze(self, expected_regime_dist: Dict[str, float], 
                observed_regime_dist: Dict[str, float]) -> Dict[str, Any]:
        """
        Analyzes regime distributions.
        expected_regime_dist: dict mapping regime name to probability (e.g. {'LOW': 0.5, 'HIGH': 0.5})
        observed_regime_dist: dict mapping regime name to probability
        """
        # Ensure identical keys
        all_keys = set(expected_regime_dist.keys()).union(set(observed_regime_dist.keys()))
        if not all_keys:
            return self._empty_result()
            
        expected_arr = np.array([expected_regime_dist.get(k, 0.0) for k in all_keys])
        observed_arr = np.array([observed_regime_dist.get(k, 0.0) for k in all_keys])
        
        # Normalize to ensure sum to 1
        expected_arr = expected_arr / (expected_arr.sum() or 1.0)
        observed_arr = observed_arr / (observed_arr.sum() or 1.0)
        
        psi = self._calc_psi(expected_arr, observed_arr)
        
        # Chi-square test (multiply by arbitrary sample size e.g. 100 for frequencies)
        chi2_stat, p_value = chisquare(f_obs=observed_arr * 100 + 1e-5, f_exp=expected_arr * 100 + 1e-5)
        
        # Calculate score
        score = 100.0
        
        if psi > self.thresholds['psi_threshold']:
            score -= (psi / self.thresholds['psi_threshold']) * 25
            
        if p_value < self.thresholds['p_value_threshold']:
            score -= 25 # Significant difference
            
        score = max(0.0, min(100.0, score))
        status = self._get_status(score)
        
        return {
            "psi": float(psi),
            "p_value": float(p_value),
            "regime_score": score,
            "status": status,
            "warnings": self._generate_warnings(psi, p_value)
        }
        
    def _calc_psi(self, expected: np.ndarray, actual: np.ndarray) -> float:
        epsilon = 1e-10
        expected_safe = expected + epsilon
        actual_safe = actual + epsilon
        
        psi_values = (actual_safe - expected_safe) * np.log(actual_safe / expected_safe)
        return float(np.sum(psi_values))

    def _get_status(self, score: float) -> str:
        if score >= 90:
            return "HEALTHY"
        elif score >= 70:
            return "MONITOR"
        elif score >= 50:
            return "WARNING"
        else:
            return "CRITICAL"
            
    def _generate_warnings(self, psi: float, p_value: float) -> List[str]:
        warnings = []
        if psi > self.thresholds['psi_threshold']:
            warnings.append(f"Regime PSI {psi:.3f} exceeds threshold {self.thresholds['psi_threshold']}")
        if p_value < self.thresholds['p_value_threshold']:
            warnings.append(f"Regime distribution differs significantly from training (p-value {p_value:.3f} < {self.thresholds['p_value_threshold']})")
        return warnings
        
    def _empty_result(self) -> Dict[str, Any]:
        return {
            "psi": 0.0,
            "p_value": 1.0,
            "regime_score": 100.0,
            "status": "HEALTHY",
            "warnings": []
        }
