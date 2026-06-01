"""
aqc/dashboard/live_dashboard.py
=================================
Main entry point for the AQC Live Paper Trading Dashboard.

Run via: `streamlit run aqc/dashboard/live_dashboard.py`

Author: Saksham Mishra — AlgoQuant Club
"""
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import streamlit as st
import pandas as pd

from aqc.live.persistence import PersistenceLayer
from aqc.dashboard.portfolio_dashboard import render_portfolio_panel
from aqc.dashboard.risk_dashboard import render_risk_panel
from aqc.dashboard.performance_dashboard import render_performance_panel

st.set_page_config(page_title="AQC Live Paper Trading", layout="wide", page_icon="📈")

# Define the DB path. Ideally from config, hardcoded here for simplicity.
DB_PATH = Path("paper_trading.db")

@st.cache_resource
def get_db():
    return PersistenceLayer(str(DB_PATH))

def main():
    st.title("AlgoQuant Club — Live Paper Trading")
    st.markdown("Real-time monitoring of strategy performance and execution on simulated/live feeds.")
    
    db = get_db()
    
    # Auto-refresh using experimental rerun or meta tag? 
    # For now, we add a manual refresh button and use st.empty() logic
    col1, col2 = st.columns([1, 10])
    with col1:
        if st.button("Refresh 🔄"):
            st.rerun()

    health = db.load_table("health_status")
    if not health.empty:
        latest_health = health.iloc[-1]
        state = latest_health["state"]
        color = "green" if state == "HEALTHY" else "orange" if state == "WARNING" else "red"
        st.markdown(f"**System Status**: <span style='color:{color}'>{state}</span> — {latest_health['message']}", unsafe_allow_html=True)
        st.markdown(f"**Latency**: {latest_health['feed_latency_ms']:.1f} ms")

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Portfolio & Equity", "Execution & Risk", "Performance Metrics"])
    
    with tab1:
        render_portfolio_panel(db)
        
    with tab2:
        render_risk_panel(db)
        
    with tab3:
        render_performance_panel(db)

if __name__ == "__main__":
    main()
