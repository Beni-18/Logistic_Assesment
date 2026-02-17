import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Telecom Churn Intelligence",
    page_icon="📡",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

body {
    background-color: #0E1117;
}

.main {
    background-color: #0E1117;
}

h1, h2, h3 {
    color: #00F5FF;
}

.stButton>button {
    background: linear-gradient(90deg, #00F5FF, #7B2FF7);
    color: white;
    font-weight: bold;
    border-radius: 12px;
    height: 3em;
    width: 100%;
}

.metric-card {
    background: linear-gradient(145deg, #111827, #1F2937);
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 0px 20px rgba(0,255,255,0.2);
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODELS ----------------
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")

# ---------------- TITLE ----------------
st.title("📡 Telecom Churn Intelligence Dashboard")
st.markdown("Advanced churn prediction using PCA + Logistic Regression")

st.markdown("---")

# ---------------- INPUT SECTION ----------------
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Customer Age", min_value=18, max_value=100, value=30)
    tenure = st.number_input("Tenure (Months)", min_value=0, max_value=120, value=12)

with col2:
    monthly_charges = st.number_input("Monthly Charges (₹)", min_value=100, max_value=10000, value=1000)
    contract = st.selectbox(
        "Contract Type",
        ["Month-to-Month", "One Year", "Two Year"]
    )

with col3:
    internet = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber"]
    )

st.markdown("---")

# ---------------- PREDICTION ----------------
if st.button("🚀 Analyze Customer Risk"):

    input_data = {
        "Age": age,
        "Tenure": tenure,
        "Monthly_Charges": monthly_charges,
        "Contract_Type": contract,
        "Internet_Service": internet
    }

    input_df = pd.DataFrame([input_data])
    input_df = pd.get_dummies(input_df)

    model_columns = [
        'Age',
        'Tenure',
        'Monthly_Charges',
        'Contract_Type_One Year',
        'Contract_Type_Two Year',
        'Internet_Service_Fiber'
    ]

    for col in model_columns:
        if col not in input_df:
            input_df[col] = 0

    input_df = input_df[model_columns]

    # Apply preprocessing
    scaled = scaler.transform(input_df)
    reduced = pca.transform(scaled)

    prediction = model.predict(reduced)[0]
    probability = model.predict_proba(reduced)[0][1]

    st.markdown("## 📊 Risk Analysis Result")

    # Probability Bar
    st.progress(float(probability))

    colA, colB = st.columns(2)

    with colA:
        st.metric("Churn Probability", f"{probability*100:.1f}%")

    with colB:
        if prediction == 1:
            st.error("⚠️ HIGH RISK – Likely to Churn")
        else:
            st.success("✅ LOW RISK – Likely to Stay")

    st.markdown("---")

    # ---------------- INSIGHTS ----------------
    st.subheader("🧠 Model Insights")

    if tenure < 12:
        st.write("• Short tenure customers tend to churn more frequently.")
    if contract == "Month-to-Month":
        st.write("• Month-to-Month contracts show higher churn behavior.")
    if monthly_charges > 1500:
        st.write("• Higher monthly charges increase churn probability.")
    if internet == "Fiber":
        st.write("• Fiber users sometimes churn due to premium pricing.")