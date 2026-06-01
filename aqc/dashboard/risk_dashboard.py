"""
aqc/dashboard/risk_dashboard.py
=================================
Streamlit component for Risk tracking.

Author: Saksham Mishra — AlgoQuant Club
"""
import streamlit as st
import pandas as pd
import plotly.express as px

def render_risk_panel(db):
    st.subheader("Execution & Risk")
    
    orders = db.load_table("orders")
    
    if orders.empty:
        st.info("No order data available yet.")
        return
        
    open_orders = orders[~orders["state"].isin(["FILLED", "CANCELLED", "REJECTED"])]
    filled_orders = orders[orders["state"] == "FILLED"]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Open Orders", len(open_orders))
        if not open_orders.empty:
            st.dataframe(open_orders[["symbol", "side", "target_qty", "state"]], hide_index=True)
            
    with col2:
        st.metric("Total Fills", len(filled_orders))
        if not filled_orders.empty:
            st.dataframe(filled_orders[["symbol", "side", "filled_qty", "avg_price"]].tail(5), hide_index=True)
            
    st.markdown("---")
    
    signals = db.load_table("signal_audit")
    if not signals.empty:
        total = len(signals)
        approved = len(signals[signals["approved"] == 1])
        rejected = total - approved
        
        st.write("### Signal Audit")
        col3, col4, col5 = st.columns(3)
        col3.metric("Total Signals", total)
        col4.metric("Approved", approved)
        col5.metric("Rejected", rejected)
        
        # Latency
        fig = px.histogram(signals, x="latency_ms", title="Signal Latency Distribution (ms)", 
                           template="plotly_dark", nbins=20, color_discrete_sequence=["#FFCA28"])
        st.plotly_chart(fig, use_container_width=True)
