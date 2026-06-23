import numpy as np
from typing import Dict, Any, List

class ForecastStabilityAnalyzer:
    """Monitors volatility forecasts and compares them to realized volatility."""
    
    def __init__(self, thresholds: Dict[str, float] = None):
        self.thresholds = thresholds or {
            "rmse_increase_threshold": 0.50, # 50% increase relative to backtest RMSE
            "mae_increase_threshold": 0.50,
            "correlation_drop": 0.40 # If correlation between forecast and realized drops below 0.4
        }
        
    def analyze(self, expected_rmse: float, expected_mae: float, 
                forecasts: np.ndarray, realized: np.ndarray) -> Dict[str, Any]:
        """
        Analyzes recent forecast accuracy against expected accuracy.
        forecasts: Array of forecasted volatility
        realized: Array of actual realized volatility over the same period
        """
        if len(forecasts) != len(realized) or len(forecasts) == 0:
            return self._empty_result()
            
        current_rmse = self._calc_rmse(forecasts, realized)
        current_mae = self._calc_mae(forecasts, realized)
        correlation = self._calc_correlation(forecasts, realized)
        
        rmse_increase = (current_rmse - expected_rmse) / expected_rmse if expected_rmse > 0 else 0
        mae_increase = (current_mae - expected_mae) / expected_mae if expected_mae > 0 else 0
        
        # Score calculation
        score = 100.0
        
        if rmse_increase > 0:
            score -= min(50, rmse_increase * 50)
        
        if correlation < 0.7:
            score -= (0.7 - max(0, correlation)) * 50
            
        score = max(0.0, min(100.0, score))
        status = self._get_status(score)
        
        return {
            "current_rmse": current_rmse,
            "current_mae": current_mae,
            "correlation": correlation,
            "rmse_increase": rmse_increase,
            "mae_increase": mae_increase,
            "forecast_score": score,
            "status": status,
            "warnings": self._generate_warnings(rmse_increase, mae_increase, correlation)
        }

    def _calc_rmse(self, forecasts: np.ndarray, realized: np.ndarray) -> float:
        return float(np.sqrt(np.mean((forecasts - realized)**2)))
        
    def _calc_mae(self, forecasts: np.ndarray, realized: np.ndarray) -> float:
        return float(np.mean(np.abs(forecasts - realized)))
        
    def _calc_correlation(self, forecasts: np.ndarray, realized: np.ndarray) -> float:
        if len(forecasts) < 2:
            return 0.0
        # Correlation matrix [0,1] is the correlation
        corr = np.corrcoef(forecasts, realized)[0, 1]
        return float(corr) if not np.isnan(corr) else 0.0

    def _get_status(self, score: float) -> str:
        if score >= 90:
            return "HEALTHY"
        elif score >= 70:
            return "MONITOR"
        elif score >= 50:
            return "WARNING"
        else:
            return "CRITICAL"
            
    def _generate_warnings(self, rmse_inc: float, mae_inc: float, corr: float) -> List[str]:
        warnings = []
        if rmse_inc > self.thresholds['rmse_increase_threshold']:
            warnings.append(f"RMSE increased by {rmse_inc:.1%} (threshold: {self.thresholds['rmse_increase_threshold']:.1%})")
        if mae_inc > self.thresholds['mae_increase_threshold']:
            warnings.append(f"MAE increased by {mae_inc:.1%} (threshold: {self.thresholds['mae_increase_threshold']:.1%})")
        if corr < self.thresholds['correlation_drop']:
            warnings.append(f"Forecast correlation {corr:.2f} below threshold {self.thresholds['correlation_drop']:.2f}")
        return warnings
        
    def _empty_result(self) -> Dict[str, Any]:
        return {
            "current_rmse": 0.0,
            "current_mae": 0.0,
            "correlation": 0.0,
            "rmse_increase": 0.0,
            "mae_increase": 0.0,
            "forecast_score": 100.0,
            "status": "HEALTHY",
            "warnings": ["Insufficient data to calculate forecast stability."]
        }
