import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# =====================================
# Load Models
# =====================================

MODEL_DIR = Path("../model")

if not MODEL_DIR.exists():
    MODEL_DIR = Path("model")

model_log = joblib.load(MODEL_DIR / "best_model_log.pkl")
model_pt = joblib.load(MODEL_DIR / "best_model_pt.pkl")
model_ss = joblib.load(MODEL_DIR / "best_model_ss.pkl")
model_mm = joblib.load(MODEL_DIR / "best_model_mm.pkl")
power_transformers = joblib.load(MODEL_DIR /"power_transformers.pkl")

# =====================================
# Page Config
# =====================================

st.set_page_config(
    page_title="Advertising Sales Prediction",
    page_icon="📈",
    layout="centered"
)

st.title("📈 Advertising Sales Prediction")

st.write("Predict product sales based on advertising budgets.")

# =====================================
# User Inputs
# =====================================

tv = st.number_input(
    "TV Advertising Budget",
    min_value=0.0,
    max_value=300.0,
    value=150.0,
    step=0.1
)

radio = st.number_input(
    "Radio Advertising Budget",
    min_value=0.0,
    max_value=50.0,
    value=25.0,
    step=0.1
)

newspaper = st.number_input(
    "Newspaper Advertising Budget",
    min_value=0.0,
    max_value=120.0,
    value=30.0,
    step=0.1
)

dominant_channel = st.selectbox(
    "Dominant Advertising Channel",
    ["TV", "Radio", "Newspaper"]
)

selected_model = st.selectbox(
    "Choose Model",
    [
        "Log Transformation",
        "Power Transformer",
        "Standard Scaler",
        "MinMax Scaler"
    ]
)

# =====================================
# Feature Engineering
# =====================================

tv_radio = tv * radio
tv_newspaper = tv * newspaper
radio_newspaper = radio * newspaper

input_df = pd.DataFrame({
    "TV": [tv],
    "Radio": [radio],
    "Newspaper": [newspaper],
    "TV_Radio": [tv_radio],
    "TV_Newspaper": [tv_newspaper],
    "Radio_Newspaper": [radio_newspaper],
    "Dominant_Channel": [dominant_channel]
})

# =====================================
# Prediction
# =====================================

if st.button("Predict Sales"):

    if selected_model == "Log Transformation":

        input_log = input_df.copy()

        log_cols = [
            "TV_Radio",
            "TV_Newspaper",
            "Radio_Newspaper"
        ]

        input_log[log_cols] = np.log1p(input_log[log_cols])

        prediction = model_log.predict(input_log)[0]

    elif selected_model == "Power Transformer":

        input_pt = input_df.copy()

        for col, transformer in power_transformers.items():
            input_pt[col] = transformer.transform(input_pt[[col]]).ravel()

        prediction = model_pt.predict(input_pt)[0]

    elif selected_model == "Standard Scaler":

        prediction = model_ss.predict(input_df)[0]

    else:

        prediction = model_mm.predict(input_df)[0]

    st.success(f"Predicted Sales: {prediction:.2f}")
    
    print(power_transformers.keys())