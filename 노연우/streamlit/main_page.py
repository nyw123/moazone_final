import streamlit as st
import pandas as pd
import pandas_profiling
import streamlit.components.v1 as components
import plotly.figure_factory as ff
import numpy as np
import mlflow
import sys

st.markdown("# Main page 🎈")
st.sidebar.markdown("# Main page 🎈")

col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", "0.6878", "0.05")
col2.metric("F1-score", "0.5974", "-0.002")
col3.metric("Total Data", "31204", "New 403")
