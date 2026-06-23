import numpy as np
from scipy.stats import entropy
from typing import Dict, Any, List

class SignalStabilityAnalyzer:
    """Analyzes signal stability, checking for frequency, direction, and distribution shifts."""
    
    def __init__(self, thresholds: Dict[str, float] = None):
        self.thresholds = thresholds or {
            "frequency_collapse": 0.50,
            "kl_divergence": 0.2, # if above 0.2, distribution has shifted significantly
            "psi_threshold": 0.25 # standard industry rule of thumb for PSI
        }
        
    def analyze(self, expected_freq: float, observed_freq: float, 
                expected_dist: np.ndarray, observed_dist: np.ndarray) -> Dict[str, Any]:
        """
        Analyzes the stability of the signal generation.
        expected_dist and observed_dist should be normalized histograms (probabilities)
        of signal values (e.g. [-1, 0, 1] or continuous confident scores binned).
        """
        freq_change = self._calc_frequency_change(expected_freq, observed_freq)
        kl_div = self._calc_kl_divergence(expected_dist, observed_dist)
        psi = self._calc_psi(expected_dist, observed_dist)
        
        # Calculate stability score
        # Base 100
        score = 100.0
        
        # Penalty for frequency collapse/explosion
        if abs(freq_change) > 0.2:
            score -= abs(freq_change) * 50 # If 50% change, drop 25 points
            
        # Penalty for distribution shifts
        if kl_div > 0.05:
            score -= (kl_div / self.thresholds['kl_divergence']) * 20
            
        if psi > 0.1:
            score -= (psi / self.thresholds['psi_threshold']) * 20
            
        score = max(0.0, min(100.0, score))
        status = self._get_status(score)
        
        return {
            "frequency_change": freq_change,
            "kl_divergence": kl_div,
            "psi": psi,
            "signal_score": score,
            "status": status,
            "warnings": self._generate_warnings(freq_change, kl_div, psi)
        }
        
    def _calc_frequency_change(self, expected: float, observed: float) -> float:
        if expected <= 0:
            return 0.0
        return (observed - expected) / expected
        
    def _calc_kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        p_safe = p + epsilon
        q_safe = q + epsilon
        return float(entropy(p_safe, q_safe))
        
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
            
    def _generate_warnings(self, freq_change: float, kl: float, psi: float) -> List[str]:
        warnings = []
        if freq_change < -self.thresholds['frequency_collapse']:
            warnings.append(f"Signal frequency collapsed by {-freq_change:.1%} (threshold: {self.thresholds['frequency_collapse']:.1%})")
        if kl > self.thresholds['kl_divergence']:
            warnings.append(f"KL Divergence {kl:.3f} exceeds threshold {self.thresholds['kl_divergence']}")
        if psi > self.thresholds['psi_threshold']:
            warnings.append(f"PSI {psi:.3f} exceeds threshold {self.thresholds['psi_threshold']}")
        return warnings
