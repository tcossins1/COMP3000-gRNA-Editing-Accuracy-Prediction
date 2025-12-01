# ui/forecas9_app.py
import sys
import os
import streamlit as st
import numpy as np
import pandas as pd
import joblib

# --- Adding /src to path to find files
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

# --- Title ---
st.title("ForeCas9: gRNA Efficiency Predictor")
st.markdown("""Enter a gRNA sequence below to predict its CRISPR-Cas9 cutting efficiency.""")

# --- User input ---
seq_input = st.text_input("Enter gRNA sequence:", "")