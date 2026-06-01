"""
aqc/dashboard/portfolio_dashboard.py
======================================
Streamlit component for Portfolio tracking.

Author: Saksham Mishra — AlgoQuant Club
"""
import streamlit as st
import pandas as pd
import plotly.express as px

def render_portfolio_panel(db):
    st.subheader("Portfolio Status")
    
    df = db.load_table("portfolio_snapshots")
    if df.empty:
        st.info("No portfolio data available yet.")
        return

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    latest = df.iloc[-1]
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Equity", f"${latest['total_equity']:,.2f}")
    col2.metric("Cash", f"${latest['cash']:,.2f}")
    col3.metric("Gross Exposure", f"${latest['gross_exposure']:,.2f}")
    col4.metric("Leverage", f"{latest['leverage']:,.2f}x")
    
    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Net Exposure", f"${latest['net_exposure']:,.2f}")
    col6.metric("Unrealised PnL", f"${latest['unrealised_pnl']:,.2f}")
    col7.metric("Realised PnL", f"${latest['realised_pnl']:,.2f}")
    col8.metric("Active Positions", int(latest['num_positions']))

    # Plot Equity Curve
    fig = px.line(df, x="timestamp", y="total_equity", title="Live Equity Curve", 
                  template="plotly_dark", color_discrete_sequence=["#00E676"])
    st.plotly_chart(fig, use_container_width=True)
