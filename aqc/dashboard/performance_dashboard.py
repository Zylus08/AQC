"""
aqc/dashboard/performance_dashboard.py
========================================
Streamlit component for Performance tracking.

Author: Saksham Mishra — AlgoQuant Club
"""
import streamlit as st
import pandas as pd
import plotly.express as px

def render_performance_panel(db):
    st.subheader("Performance Analytics")
    
    df = db.load_table("performance_metrics")
    if df.empty:
        st.info("No performance data available yet.")
        return

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    latest = df.iloc[-1]
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Live Sharpe", f"{latest['sharpe']:.2f}")
    col2.metric("Sortino", f"{latest['sortino']:.2f}")
    col3.metric("Max Drawdown", f"{latest['max_drawdown']*100:.2f}%")
    col4.metric("CAGR", f"{latest['cagr']*100:.2f}%")
    
    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Daily Return", f"{latest['daily_return']*100:.2f}%")
    col6.metric("Win Rate", f"{latest['win_rate']*100:.1f}%")
    col7.metric("Profit Factor", f"{latest['profit_factor']:.2f}")
    
    # Plot rolling sharpe
    if len(df) > 1:
        fig = px.line(df, x="timestamp", y="sharpe", title="Rolling Sharpe Ratio", 
                      template="plotly_dark", color_discrete_sequence=["#29B6F6"])
        st.plotly_chart(fig, use_container_width=True)
