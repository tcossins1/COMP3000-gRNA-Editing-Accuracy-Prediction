import sys
import os
import streamlit as st
import numpy as np
import joblib

# --- Adding /src to path to find files
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from feature_extraction.extract_features_v1 import extract_features_from_sequence

# --- Load trained model ---
model_path = os.path.join(os.path.dirname(__file__), "../models/linear_regression_v1.joblib")
model = joblib.load(model_path)

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="ForeCas9 Predictor",
    layout="centered"
)

# --- TITLE ---
st.markdown("<h1 style='text-align:center; margin-bottom: -10px;'>ForeCas9</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:18px;'>gRNA Cutting Efficiency Predictor</p>", unsafe_allow_html=True)

st.write("")

# --- INPUT CARD ---
with st.container():
    st.markdown("### Enter Your gRNA Sequence")
    st.markdown("<p style='color: grey;'>Must be 20 nucleotides long (A, T, G, C).</p>", unsafe_allow_html=True)
    seq_input = st.text_input("", placeholder="e.g., ATGCGTAGCTAAGCTAGCAC")

# --- PROCESS INPUT ---
if seq_input:
    seq = seq_input.strip().upper()
    valid_chars = set("ATGC")

    if len(seq) != 20:
        st.error("Sequence must be exactly **20 nucleotides**.")
    elif not set(seq).issubset(valid_chars):
        st.error("Sequence must contain only **A, T, G, C**.")
    else:
        # --- Extract features using backend function ---
        features = extract_features_from_sequence(seq)
        X = np.array([[features["gc_content"]]])  # 2D array for sklearn

        # --- Predict ---
        prediction = model.predict(X)[0]

        # --- RESULT CARD ---
        st.write("")
        st.markdown("### Prediction Result")
        st.subheader("Predicted Efficiency")
        st.metric(label="Efficiency Score", value=f"{prediction:.4f}")
        st.markdown(f"**GC Content:** {features['gc_content']:.2f}")
