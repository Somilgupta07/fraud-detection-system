import streamlit as st
import joblib
import pandas as pd
import numpy as np

# -----------------------------
# Load Model and Column Structure
# -----------------------------
model = joblib.load("fraud_model.pkl")
model_columns = joblib.load("model_columns.pkl")

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="💳",
    layout="centered"
)

# -----------------------------
# Title
# -----------------------------
st.title("💳 Credit Card Fraud Detection")
st.write("Enter transaction details to check whether it is fraudulent.")

st.divider()

# -----------------------------
# User Inputs
# -----------------------------
st.subheader("Transaction Information")

amt = st.number_input("Transaction Amount ($)", min_value=0.0)

category = st.number_input("Transaction Category (Encoded Value)")

lat = st.number_input("Customer Latitude")

long = st.number_input("Customer Longitude")

merch_lat = st.number_input("Merchant Latitude")

merch_long = st.number_input("Merchant Longitude")

st.divider()

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("🔍 Detect Fraud"):

    # Create input dictionary
    input_data = {
        "amt": amt,
        "category": category,
        "lat": lat,
        "long": long,
        "merch_lat": merch_lat,
        "merch_long": merch_long
    }

    # Convert to DataFrame
    df = pd.DataFrame([input_data])

    # Add missing columns (required by model)
    for col in model_columns:
        if col not in df.columns:
            df[col] = 0

    # Arrange columns in same order as training
    df = df[model_columns]

    # Prediction
    prediction = model.predict(df)[0]

    # Probability
    probability = model.predict_proba(df)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("🚨 Fraudulent Transaction Detected!")
    else:
        st.success("✅ Legitimate Transaction")

    st.write(f"Fraud Probability: **{probability:.2f}**")

    # Risk meter
    if probability > 0.8:
        st.warning("⚠️ High Risk Transaction")
    elif probability > 0.4:
        st.info("⚠️ Medium Risk")
    else:
        st.success("Low Risk Transaction")

st.divider()

# -----------------------------
# Footer
# -----------------------------
st.caption("Machine Learning Fraud Detection System")