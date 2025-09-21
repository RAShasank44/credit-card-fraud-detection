import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ===============================
# Load Model
# ===============================
@st.cache_resource
def load_model():
    return joblib.load("xgb_fraud_model.pkl")

model = load_model()

# ===============================
# Streamlit App
# ===============================
st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")

st.title("💳 Credit Card Fraud Detection")
st.write("Enter transaction details to predict whether it's **Fraudulent (1)** or **Normal (0)**.")

# Sidebar Info
st.sidebar.title("About")
st.sidebar.info(
    "This app uses an **XGBoost model** trained on the Credit Card Fraud dataset.\n\n"
    "Upload data or enter transaction details manually to get predictions."
)

# ===============================
# Input Options
# ===============================
option = st.radio("Choose input method:", ["Manual Input", "Upload CSV"])

if option == "Manual Input":
    st.subheader("Enter Transaction Features")
    # Example: Amount, V1, V2, ..., V28
    amount = st.number_input("Amount", min_value=0.0, value=100.0, step=10.0)
    time = st.number_input("Time", min_value=0.0, value=50000.0, step=1000.0)

    # Add a few PCA components (V1..V5 for demo, in practice include all features)
    v1 = st.number_input("V1", value=0.0)
    v2 = st.number_input("V2", value=0.0)
    v3 = st.number_input("V3", value=0.0)
    v4 = st.number_input("V4", value=0.0)
    v5 = st.number_input("V5", value=0.0)

    input_data = pd.DataFrame([[time, v1, v2, v3, v4, v5, amount]],
                              columns=["Time", "V1", "V2", "V3", "V4", "V5", "Amount"])

    if st.button("Predict"):
        prediction = model.predict(input_data)[0]
        st.success(f"Prediction: {'🚨 Fraud' if prediction==1 else '✅ Normal'}")

elif option == "Upload CSV":
    st.subheader("Upload CSV File")
    file = st.file_uploader("Upload your CSV file", type=["csv"])

    if file is not None:
        data = pd.read_csv(file)
        st.write("Uploaded Data Preview:", data.head())

        if st.button("Predict on Uploaded Data"):
            preds = model.predict(data)
            data["Prediction"] = preds
            st.write("Predictions:")
            st.dataframe(data)
            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions", data=csv, file_name="predictions.csv", mime="text/csv")
