from typing import Dict, Any, List
import numpy as np

from .expectation_tracker import ExpectationTracker, ExpectedPerformanceProfile
from .alpha_decay import AlphaDecayAnalyzer
from .signal_stability import SignalStabilityAnalyzer
from .forecast_stability import ForecastStabilityAnalyzer
from .regime_drift import RegimeDriftAnalyzer
from .execution_validation import ExecutionValidator
from .validation_engine import StrategyHealthEngine
from .model_governance import ModelGovernanceEngine

class ForwardValidationFramework:
    """Orchestrates the entire forward validation process."""
    
    def __init__(self):
        self.expectation_tracker = ExpectationTracker()
        self.alpha_analyzer = AlphaDecayAnalyzer()
        self.signal_analyzer = SignalStabilityAnalyzer()
        self.forecast_analyzer = ForecastStabilityAnalyzer()
        self.regime_analyzer = RegimeDriftAnalyzer()
        self.execution_validator = ExecutionValidator()
        self.health_engine = StrategyHealthEngine()
        self.governance_engine = ModelGovernanceEngine()
        
    def add_expected_profile(self, strategy_id: str, profile: ExpectedPerformanceProfile) -> None:
        self.expectation_tracker.add_profile(strategy_id, profile)
        
    def register_model(self, version_id: str, strategy_name: str, expected_profile_id: str) -> None:
        self.governance_engine.register_model(version_id, strategy_name, expected_profile_id)
        
    def validate(self, version_id: str, live_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs the full validation suite for a given model version.
        
        Args:
            version_id: The registered model version ID.
            live_metrics: Dictionary containing required data for all components:
                - alpha: dict with sharpe, cagr, profit_factor, win_rate
                - signal: dict with freq, dist (numpy array)
                - forecast: dict with forecasts (array), realized (array)
                - regime: dict with dist (dict mapping regime names to probabilities)
                - execution: dict with slippage, fill_rate, cost
                
        Returns:
            A comprehensive validation report.
        """
        model_history = self.governance_engine.models.get(version_id)
        if not model_history:
            raise ValueError(f"Model {version_id} not registered in Governance Engine.")
            
        profile_id = model_history.expected_profile_id
        expected = self.expectation_tracker.get_profile(profile_id)
        
        # 1. Alpha Decay
        alpha_res = self.alpha_analyzer.analyze(expected, live_metrics.get('alpha', {}))
        
        # 2. Signal Stability
        signal_data = live_metrics.get('signal', {})
        signal_res = self.signal_analyzer.analyze(
            expected_freq=expected.signal_frequency,
            observed_freq=signal_data.get('freq', 0),
            expected_dist=expected.metadata.get('signal_dist', np.array([1.0])),
            observed_dist=signal_data.get('dist', np.array([1.0]))
        )
        
        # 3. Forecast Stability
        forecast_data = live_metrics.get('forecast', {})
        forecast_res = self.forecast_analyzer.analyze(
            expected_rmse=expected.forecast_accuracy,
            expected_mae=expected.metadata.get('expected_mae', expected.forecast_accuracy),
            forecasts=forecast_data.get('forecasts', np.array([])),
            realized=forecast_data.get('realized', np.array([]))
        )
        
        # 4. Regime Drift
        regime_data = live_metrics.get('regime', {})
        regime_res = self.regime_analyzer.analyze(
            expected_regime_dist=expected.metadata.get('regime_dist', {}),
            observed_regime_dist=regime_data.get('dist', {})
        )
        
        # 5. Execution Validation
        exec_data = live_metrics.get('execution', {})
        exec_res = self.execution_validator.analyze(
            expected_slippage=expected.metadata.get('expected_slippage', 0.0),
            observed_slippage=exec_data.get('slippage', 0.0),
            expected_fill_rate=expected.metadata.get('expected_fill_rate', 1.0),
            observed_fill_rate=exec_data.get('fill_rate', 1.0),
            expected_cost=expected.metadata.get('expected_cost', 0.0),
            observed_cost=exec_data.get('cost', 0.0)
        )
        
        # 6. Overall Health
        health_report = self.health_engine.generate_report(
            alpha_res, signal_res, forecast_res, regime_res, exec_res
        )
        
        # Store comprehensive report
        full_report = {
            "version_id": version_id,
            "timestamp": None, # Will be set by governance engine implicitly in history
            "alpha": alpha_res,
            "signal": signal_res,
            "forecast": forecast_res,
            "regime": regime_res,
            "execution": exec_res,
            "health": health_report
        }
        
        # Update Governance
        self.governance_engine.update_health(version_id, health_report)
        
        # 7. Retraining Advisor
        recommendation = self.governance_engine.advise_retraining(
            version_id,
            current_sharpe=live_metrics.get('alpha', {}).get('sharpe', 0.0),
            expected_sharpe=expected.sharpe
        )
        
        full_report['retraining_recommendation'] = recommendation
        
        return full_report
