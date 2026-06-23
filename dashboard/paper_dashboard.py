import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Paper Trading Dashboard", layout="wide")
st.title("Paper Trading Analytics (20-60 Days)")

col1, col2, col3 = st.columns(3)
col1.metric("Days Active", "45", "ON TRACK")
col2.metric("Paper CAGR", "18.5%", "-2.0% vs expected")
col3.metric("Paper Sharpe", "1.6", "-0.2 vs expected")

st.header("Rolling Performance")
df = pd.DataFrame(
    np.cumsum(np.random.randn(45, 1) * 0.01 + 0.001),
    columns=['Cumulative Return']
)
st.line_chart(df)

st.header("Survival Probability")
st.progress(0.85, text="85% Probability of Survival")
