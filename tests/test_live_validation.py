import pytest
import numpy as np
from datetime import datetime
import os

from aqc.live_validation.expectation_tracker import ExpectationTracker, ExpectedPerformanceProfile
from aqc.live_validation.alpha_decay import AlphaDecayAnalyzer
from aqc.live_validation.signal_stability import SignalStabilityAnalyzer
from aqc.live_validation.forecast_stability import ForecastStabilityAnalyzer
from aqc.live_validation.regime_drift import RegimeDriftAnalyzer
from aqc.live_validation.execution_validation import ExecutionValidator
from aqc.live_validation.validation_engine import StrategyHealthEngine, EarlyWarningSystem
from aqc.live_validation.model_governance import ModelGovernanceEngine
from aqc.live_validation.forward_validation import ForwardValidationFramework
from aqc.live_validation.validation_reports import ValidationReports

@pytest.fixture
def expected_profile():
    return ExpectedPerformanceProfile(
        sharpe=2.0,
        sortino=2.5,
        cagr=0.20,
        win_rate=0.55,
        profit_factor=1.5,
        max_drawdown=0.10,
        signal_frequency=100.0,
        forecast_accuracy=0.02,
        sharpe_range=(1.8, 2.2),
        win_rate_range=(0.50, 0.60),
        profit_factor_range=(1.3, 1.7),
        signal_frequency_range=(90, 110),
        metadata={
            "expected_mae": 0.015,
            "regime_dist": {"LOW": 0.3, "NORMAL": 0.5, "HIGH": 0.2},
            "expected_slippage": 0.0005,
            "expected_fill_rate": 0.98,
            "expected_cost": 0.001
        }
    )

def test_alpha_decay_analyzer(expected_profile):
    analyzer = AlphaDecayAnalyzer()
    
    # Healthy case
    res = analyzer.analyze(expected_profile, {"sharpe": 2.1, "cagr": 0.22, "win_rate": 0.56, "profit_factor": 1.6})
    assert res["status"] == "HEALTHY"
    assert res["decay_score"] == 100
    
    # Degraded case
    res = analyzer.analyze(expected_profile, {"sharpe": 1.0, "cagr": 0.10, "win_rate": 0.45, "profit_factor": 1.1})
    assert res["decay_score"] < 100
    assert len(res["warnings"]) > 0

def test_signal_stability_analyzer():
    analyzer = SignalStabilityAnalyzer()
    
    expected_dist = np.array([0.2, 0.6, 0.2])
    observed_dist = np.array([0.2, 0.6, 0.2])
    
    # Healthy
    res = analyzer.analyze(100.0, 95.0, expected_dist, observed_dist)
    assert res["status"] == "HEALTHY"
    
    # Collapsed freq
    res = analyzer.analyze(100.0, 20.0, expected_dist, observed_dist)
    assert res["frequency_change"] == -0.8
    assert len(res["warnings"]) > 0
    
    # Distribution shift
    shifted_dist = np.array([0.8, 0.1, 0.1])
    res = analyzer.analyze(100.0, 100.0, expected_dist, shifted_dist)
    assert res["kl_divergence"] > 0
    assert res["status"] != "HEALTHY"

def test_forecast_stability_analyzer():
    analyzer = ForecastStabilityAnalyzer()
    
    forecasts = np.array([0.02, 0.021, 0.019, 0.022, 0.02])
    realized = np.array([0.02, 0.021, 0.019, 0.022, 0.02])
    
    # Perfect match
    res = analyzer.analyze(0.02, 0.015, forecasts, realized)
    assert res["status"] == "HEALTHY"
    assert res["rmse_increase"] <= 0
    
    # Degraded match
    realized_bad = np.array([0.06, 0.061, 0.059, 0.062, 0.06])
    res = analyzer.analyze(0.02, 0.015, forecasts, realized_bad)
    assert res["status"] != "HEALTHY"
    assert res["rmse_increase"] > 0

def test_regime_drift_analyzer():
    analyzer = RegimeDriftAnalyzer()
    
    expected = {"LOW": 0.3, "NORMAL": 0.5, "HIGH": 0.2}
    
    # Same
    res = analyzer.analyze(expected, expected)
    assert res["status"] == "HEALTHY"
    
    # Shifted
    shifted = {"LOW": 0.8, "NORMAL": 0.1, "HIGH": 0.1}
    res = analyzer.analyze(expected, shifted)
    assert res["psi"] > 0.1
    assert len(res["warnings"]) > 0

def test_execution_validator():
    validator = ExecutionValidator()
    
    # Healthy
    res = validator.analyze(0.0005, 0.0004, 0.98, 0.99, 0.001, 0.0009)
    assert res["status"] == "HEALTHY"
    
    # Degraded
    res = validator.analyze(0.0005, 0.0010, 0.98, 0.80, 0.001, 0.0015)
    assert res["status"] != "HEALTHY"
    assert res["slippage_change"] == 1.0 # 100% worse

def test_model_governance(expected_profile):
    engine = ModelGovernanceEngine()
    
    engine.register_model("v1.0", "TestStrategy", "Profile1")
    assert "v1.0" in engine.models
    
    health_report = {"overall_score": 45.0, "status": "CRITICAL"}
    engine.update_health("v1.0", health_report)
    
    history = engine.get_model_history("v1.0")
    assert len(history) == 1
    assert history[0]["status"] == "CRITICAL"
    
    registry = engine.generate_registry_report()
    assert len(registry) == 1
    assert registry[0]["latest_health_score"] == 45.0
    
    rec = engine.advise_retraining("v1.0", 0.5, 2.0)
    assert rec["recommendation"] == "RETRAIN MODEL"

def test_forward_validation_framework(expected_profile):
    fw = ForwardValidationFramework()
    
    fw.add_expected_profile("Profile1", expected_profile)
    fw.register_model("v1.0", "TestStrategy", "Profile1")
    
    live_metrics = {
        "alpha": {"sharpe": 2.1, "cagr": 0.22, "win_rate": 0.56, "profit_factor": 1.6},
        "signal": {"freq": 98.0, "dist": np.array([1.0])},
        "forecast": {"forecasts": np.array([0.02]), "realized": np.array([0.02])},
        "regime": {"dist": {"LOW": 0.3, "NORMAL": 0.5, "HIGH": 0.2}},
        "execution": {"slippage": 0.0005, "fill_rate": 0.98, "cost": 0.001}
    }
    
    report = fw.validate("v1.0", live_metrics)
    assert report["health"]["status"] == "HEALTHY"
    assert "retraining_recommendation" in report

def test_validation_reports(tmp_path):
    reports = ValidationReports(output_dir=str(tmp_path), plots_dir=str(tmp_path))
    
    data = [{"timestamp": "2023-01-01", "sharpe_decay": 0.1, "overall_score": 90}]
    
    reports.export_alpha_decay_report(data)
    assert os.path.exists(os.path.join(tmp_path, "alpha_decay_report.csv"))
    
    reports.plot_alpha_decay_curve(data)
    assert os.path.exists(os.path.join(tmp_path, "alpha_decay_curve.png"))
    
    reports.plot_health_score_timeline(data)
    assert os.path.exists(os.path.join(tmp_path, "health_score_timeline.png"))
