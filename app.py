import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ===============================
# Load Model
# ===============================
@st.cache_resource
def load_model():
    return joblib.load("xgb_fraud_model.pkl")  # path in repo

model = load_model()

st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")
st.title("💳 Credit Card Fraud Detection")
st.write("Predict whether a credit card transaction is **Fraudulent (1)** or **Normal (0)**.")

# ===============================
# Tabs for Single & Bulk Prediction
# ===============================
tab1, tab2 = st.tabs(["🔹 Single Transaction", "📂 Bulk CSV Prediction"])

# ---------------------------
# Single Transaction Prediction
# ---------------------------
with tab1:
    st.subheader("Manual Transaction Input")

    # Input fields for demo (you can extend to all 30 features)
    time = st.number_input("Time", value=1000.0)
    amount = st.number_input("Amount", value=100.0)
    v1 = st.number_input("V1", value=0.0)
    v2 = st.number_input("V2", value=0.0)
    v3 = st.number_input("V3", value=0.0)
    v4 = st.number_input("V4", value=0.0)
    v5 = st.number_input("V5", value=0.0)

    features = pd.DataFrame([[time, v1, v2, v3, v4, v5, amount]],
                            columns=["Time", "V1", "V2", "V3", "V4", "V5", "Amount"])

    if st.button("Predict Transaction"):
        pred = model.predict(features)[0]
        st.success(f"Prediction: {'🚨 Fraud' if pred==1 else '✅ Normal'}")

        # Optional: SHAP explanation
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(features)
            st.subheader("Feature Importance (SHAP)")
            shap.initjs()
            plt.figure()
            shap.summary_plot(shap_values, features, plot_type="bar", show=False)
            st.pyplot(plt)
        except Exception as e:
            st.warning("SHAP explanation not available.")

# ---------------------------
# Bulk CSV Prediction
# ---------------------------
with tab2:
    st.subheader("Upload CSV for Bulk Prediction")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:", df.head())

        # ✅ Drop Class column if it exists
        if 'Class' in df.columns:
            X_input = df.drop('Class', axis=1)
        else:
            X_input = df.copy()

        # Make predictions
        preds = model.predict(X_input)
        df['Prediction'] = preds
        st.write("Prediction Results:")
        st.dataframe(df)

        # Download button
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions CSV", data=csv, file_name="predictions.csv", mime="text/csv")

        # Optional: simple bar chart of predictions
        summary = df['Prediction'].value_counts()
        st.bar_chart(summary)
