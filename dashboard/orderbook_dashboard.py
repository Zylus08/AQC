import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Order Book Dashboard", layout="wide")
st.title("Order Book Imbalance & Dynamics")

col1, col2, col3 = st.columns(3)
col1.metric("Top Level Imbalance", "+0.45", "Buy Heavy")
col2.metric("Microprice Deviation", "+0.5 bps", "Bullish")
col3.metric("Book Pressure", "2.1", "High")

st.header("Imbalance Over Time")
df = pd.DataFrame(
    np.random.randn(50, 1) * 0.2 + 0.1,
    columns=['Order Book Imbalance']
)
st.area_chart(df)

st.header("Predictive Signals")
st.write("Next Tick Direction Probability:")
st.progress(0.65, text="65% UP")
