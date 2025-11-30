import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="AI Traffic Control – Results", layout="wide")
st.title("🚦 AI vs Fixed-Time – SUMO Results (Precomputed)")

fixed_returns = np.load("fixed_returns.npy")
ai_returns = np.load("ai_returns.npy")
fixed_trace = np.load("fixed_trace_sample.npy")
ai_trace = np.load("ai_trace_sample.npy")
summary = pd.read_csv("results_summary.csv")

fixed_wait = -fixed_returns * 1000
ai_wait = -ai_returns * 1000

mean_fixed = fixed_wait.mean()
mean_ai = ai_wait.mean()
improvement = (mean_fixed - mean_ai) / mean_fixed * 100

c1, c2, c3 = st.columns(3)
c1.metric("Fixed-Time avg waiting index", f"{mean_fixed:,.0f}")
c2.metric("AI avg waiting index", f"{mean_ai:,.0f}")
c3.metric("AI improvement", f"{improvement:.1f}%")

st.markdown("---")
st.subheader("Episode returns")
st.bar_chart({"Fixed-time": fixed_returns, "AI (A2C+GNN)": ai_returns})

st.subheader("Reward / waiting over time (sample episode)")
st.line_chart({"Fixed-time": fixed_trace, "AI (A2C+GNN)": ai_trace})

st.subheader("Per-episode summary")
st.dataframe(summary)
