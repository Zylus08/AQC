import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Microstructure Dashboard", layout="wide")
st.title("Market Microstructure")

col1, col2, col3, col4 = st.columns(4)
col1.metric("VPIN (Toxicity)", "0.75", "High")
col2.metric("Variance Ratio", "1.12", "Trending")
col3.metric("Adverse Selection", "0.8 bps", "+0.1 bps")
col4.metric("OFI", "0.2", "Net Buy")

st.header("Flow Toxicity (VPIN)")
df = pd.DataFrame(
    np.random.randn(30, 1) * 0.1 + 0.6,
    columns=['VPIN']
)
st.line_chart(df)

st.header("Adverse Selection Markouts")
markouts = pd.DataFrame({
    'Horizon': ['1s', '5s', '10s', '30s', '60s'],
    'Markout (bps)': [-0.1, -0.3, -0.5, -0.7, -0.8]
})
st.bar_chart(markouts.set_index('Horizon'))
