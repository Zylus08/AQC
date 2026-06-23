import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Execution Dashboard", layout="wide")
st.title("Execution Quality Dashboard")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Average Slippage", "0.5 bps", "-0.1 bps")
col2.metric("Market Impact", "1.2 bps", "+0.2 bps")
col3.metric("Fill Rate", "98.5%", "+0.5%")
col4.metric("Execution Cost", "$1,250", "-$50")

st.header("Execution Costs Over Time")
# Mock data
df = pd.DataFrame(
    np.random.randn(20, 2) * [0.5, 0.2] + [1.5, 0.5],
    columns=['Market Impact (bps)', 'Slippage (bps)']
)
st.line_chart(df)

st.header("Liquidity Profile")
liquidity_data = pd.DataFrame({
    'Regime': ['NORMAL', 'THIN', 'STRESS', 'CRISIS'],
    'Frequency': [60, 25, 10, 5]
})
st.bar_chart(liquidity_data.set_index('Regime'))
