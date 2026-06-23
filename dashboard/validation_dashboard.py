import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path to allow importing aqc
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aqc.live_validation.expectation_tracker import ExpectedPerformanceProfile
from aqc.live_validation.forward_validation import ForwardValidationFramework

st.set_page_config(page_title="Forward Validation Framework", layout="wide")

st.title("Forward Validation Framework Dashboard")

# Initialize Framework (Mocking data for dashboard display)
@st.cache_resource
def get_framework():
    fw = ForwardValidationFramework()
    
    # Create mock expectation
    profile = ExpectedPerformanceProfile(
        sharpe=1.85,
        sortino=2.1,
        cagr=0.25,
        win_rate=0.55,
        profit_factor=1.4,
        max_drawdown=0.15,
        signal_frequency=100.0,
        forecast_accuracy=0.015,
        sharpe_range=(1.5, 2.2),
        win_rate_range=(0.5, 0.6),
        profit_factor_range=(1.2, 1.6),
        signal_frequency_range=(80, 120),
        metadata={
            "expected_mae": 0.01,
            "regime_dist": {"NORMAL": 0.6, "HIGH": 0.3, "EXTREME": 0.1},
            "expected_slippage": 0.0005,
            "expected_fill_rate": 0.98,
            "expected_cost": 0.001
        }
    )
    fw.add_expected_profile("StrategyA_Profile", profile)
    fw.register_model("v1.0", "StrategyA", "StrategyA_Profile")
    return fw

fw = get_framework()

# Simulate live metrics based on user input for demonstration
st.sidebar.header("Simulate Live Metrics")
live_sharpe = st.sidebar.slider("Live Sharpe", 0.0, 3.0, 1.1)
live_cagr = st.sidebar.slider("Live CAGR", -0.5, 1.0, 0.15)
live_win_rate = st.sidebar.slider("Live Win Rate", 0.0, 1.0, 0.48)
live_profit_factor = st.sidebar.slider("Live Profit Factor", 0.0, 3.0, 1.1)

live_signal_freq = st.sidebar.slider("Live Signal Frequency", 0.0, 200.0, 42.0)
live_rmse = st.sidebar.slider("Live Forecast RMSE", 0.0, 0.1, 0.035)

live_metrics = {
    "alpha": {
        "sharpe": live_sharpe,
        "cagr": live_cagr,
        "win_rate": live_win_rate,
        "profit_factor": live_profit_factor
    },
    "signal": {
        "freq": live_signal_freq,
        "dist": np.array([0.2, 0.6, 0.2])  # Mock changed distribution
    },
    "forecast": {
        "forecasts": np.random.normal(0.02, 0.005, 100),
        "realized": np.random.normal(0.025, 0.008, 100) # Slightly higher realized
    },
    "regime": {
        "dist": {"NORMAL": 0.4, "HIGH": 0.5, "EXTREME": 0.1}
    },
    "execution": {
        "slippage": 0.0008,
        "fill_rate": 0.95,
        "cost": 0.0012
    }
}

report = fw.validate("v1.0", live_metrics)

# --- Strategy Health Section ---
st.header("Strategy Health")
health = report['health']

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Overall Score", f"{health['overall_score']:.1f}", health['status'])
col2.metric("Alpha Score", f"{health['component_scores']['alpha_score']:.1f}")
col3.metric("Signal Score", f"{health['component_scores']['signal_score']:.1f}")
col4.metric("Forecast Score", f"{health['component_scores']['forecast_score']:.1f}")
col5.metric("Regime Score", f"{health['component_scores']['regime_score']:.1f}")
col6.metric("Execution Score", f"{health['component_scores']['execution_score']:.1f}")

st.markdown("---")

# --- Alpha Decay Section ---
st.header("Alpha Decay")
alpha = report['alpha']
col1, col2, col3 = st.columns(3)
col1.metric("Sharpe Decay", f"{alpha['sharpe_decay']:.1%}")
col2.metric("Return Decay", f"{alpha['return_decay']:.1%}")
col3.metric("Win Rate Decay", f"{alpha['win_rate_decay']:.1%}")

st.markdown("---")

# --- Signal Stability Section ---
st.header("Signal Stability")
signal = report['signal']
col1, col2 = st.columns(2)
col1.metric("Frequency Change", f"{signal['frequency_change']:.1%}")
col2.metric("KL Divergence", f"{signal['kl_divergence']:.3f}")

st.markdown("---")

# --- Forecast Accuracy Section ---
st.header("Forecast Accuracy")
forecast = report['forecast']
col1, col2 = st.columns(2)
col1.metric("RMSE Increase", f"{forecast['rmse_increase']:.1%}")
col2.metric("MAE Increase", f"{forecast['mae_increase']:.1%}")

st.markdown("---")

# --- Regime Drift Section ---
st.header("Regime Drift")
regime = report['regime']
col1, col2 = st.columns(2)
col1.metric("Regime PSI", f"{regime['psi']:.3f}")
col2.metric("P-Value", f"{regime['p_value']:.3f}")

st.markdown("---")

# --- Execution Quality Section ---
st.header("Execution Quality")
execution = report['execution']
col1, col2, col3 = st.columns(3)
col1.metric("Slippage Change", f"{execution['slippage_change']:.1%}")
col2.metric("Fill Rate Change", f"{execution['fill_rate_change']:.1%}")
col3.metric("Cost Change", f"{execution['cost_change']:.1%}")

st.markdown("---")

# --- Alerts Section ---
st.header("Alerts & Recommendations")

retraining = report['retraining_recommendation']
st.info(f"**Retraining Recommendation:** {retraining['recommendation']} (Confidence: {retraining['confidence']}%) - {retraining['reason']}")

alerts = health['alerts']
if not alerts:
    st.success("No active warnings.")
else:
    for alert in alerts:
        if alert['level'] == 'CRITICAL':
            st.error(f"[{alert['category']}] {alert['message']}")
        else:
            st.warning(f"[{alert['category']}] {alert['message']}")
