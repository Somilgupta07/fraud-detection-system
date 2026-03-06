import streamlit as st
import joblib
import pandas as pd
import numpy as np

# -----------------------------
# 1. Page Configuration
# -----------------------------
st.set_page_config(
    page_title="SafeGuard | Fraud Detection",
    page_icon="🛡️",
    layout="centered"
)

# Custom CSS for a cleaner, modern look
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    div.stButton > button:first-child {
        background-color: #007bff;
        color: white;
        height: 3em;
        width: 100%;
        border-radius: 8px;
        font-weight: bold;
    }
    .result-card {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------
# 2. Optimized Asset Loading
# -----------------------------
@st.cache_resource
def load_assets():
    try:
        model = joblib.load("fraud_model.pkl")
        model_columns = joblib.load("model_columns.pkl")
        return model, model_columns
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None

model, model_columns = load_assets()

# -----------------------------
# 3. Header & Sidebar
# -----------------------------
st.title("🛡️ SafeGuard AI")
st.subheader("Professional Fraud Analysis System")
st.write("Evaluate transaction safety using real-time machine learning patterns.")

with st.sidebar:
    st.header("About System")
    st.info("This system analyzes transaction amount, category, and geolocation to determine risk probability.")
    st.markdown("---")
    st.caption("v1.2.0 Build | Stable")

# -----------------------------
# 4. Input Form (Modern Layout)
# -----------------------------
st.markdown("### 📝 Transaction Entry")

# Organized into two main columns
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Financial Details**")
    amt = st.number_input("Transaction Amount ($)", min_value=0.0, step=10.0, help="The total value of the transaction")
    category = st.number_input("Category Code", min_value=0, step=1, help="Numeric encoding for the merchant category")

with col2:
    st.markdown("**User Geolocation**")
    lat = st.number_input("Customer Latitude", value=0.0, format="%.6f")
    long = st.number_input("Customer Longitude", value=0.0, format="%.6f")

st.markdown("**Merchant Geolocation**")
m_col1, m_col2 = st.columns(2)
with m_col1:
    merch_lat = st.number_input("Merchant Latitude", value=0.0, format="%.6f")
with m_col2:
    merch_long = st.number_input("Merchant Longitude", value=0.0, format="%.6f")

st.divider()

# -----------------------------
# 5. Prediction Logic
# -----------------------------
if st.button("🚀 Analyze Transaction Risk"):
    if model is None:
        st.error("Model not loaded. Analysis cannot proceed.")
    else:
        # Create input dictionary
        input_data = {
            "amt": amt,
            "category": category,
            "lat": lat,
            "long": long,
            "merch_lat": merch_lat,
            "merch_long": merch_long
        }

        # Processing Data
        df = pd.DataFrame([input_data])
        for col in model_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[model_columns]

        # Model Inference
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

        # -----------------------------
        # 6. Result Display (Modern Dashboard Style)
        # -----------------------------
        st.markdown("### 📊 Analysis Result")
        
        # Determine Status Colors
        if prediction == 1:
            status_text = "FRAUD DETECTED"
            status_color = "red"
            st.error(f"### 🚨 ALERT: {status_text}")
        else:
            status_text = "LEGITIMATE"
            status_color = "green"
            st.success(f"### ✅ STATUS: {status_text}")

        # Metrics Row
        r1, r2 = st.columns(2)
        r1.metric(label="Risk Probability", value=f"{probability*100:.2f}%")
        
        # Risk Description
        if probability > 0.8:
            r2.warning("Level: High Risk")
        elif probability > 0.4:
            r2.info("Level: Medium Risk")
        else:
            r2.success("Level: Low Risk")

        # Visual Risk Bar
        st.progress(float(probability))

# -----------------------------
# 7. Footer
# -----------------------------
st.divider()
st.caption("This system provides probability estimates based on historical data. Use as a decision-support tool.")