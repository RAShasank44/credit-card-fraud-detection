import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")
st.title("💳 Credit Card Fraud Detection (Flexible Input)")

# Load Model with Error Handling
@st.cache_resource
def load_model(path="xgb_fraud_model.pkl"):
    if not os.path.exists(path):
        st.error(f"❌ Model file not found at `{path}`")
        return None
    return joblib.load(path)

model = load_model()
if model is None:
    st.stop()

# Get model expected feature names
expected_features = model.get_booster().feature_names


# Tabs for Single & Bulk Prediction
tab1, tab2 = st.tabs(["🔹 Single Transaction", "📂 Bulk CSV Prediction"])

# Single Transaction Prediction
with tab1:
    st.subheader("Manual Transaction Input")

    user_input = {}
    for feat in expected_features:
        user_input[feat] = st.number_input(feat, value=0.0)

    input_df = pd.DataFrame([user_input])

    if st.button("Predict Transaction"):
        # Align features
        for col in expected_features:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[expected_features]

        pred = model.predict(input_df)[0]
        st.success(f"Prediction: {'🚨 Fraud' if pred==1 else '✅ Normal'}")


# Bulk CSV Prediction
with tab2:
    st.subheader("Upload CSV for Bulk Prediction")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:", df.head())

        # Drop target column if exists
        if 'Class' in df.columns:
            X_input = df.drop('Class', axis=1)
        else:
            X_input = df.copy()

        try:
            preds = model.predict(X_input)
            df['Prediction'] = preds
            st.write("Prediction Results:")
            st.dataframe(df)

            # Download predictions
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions CSV", data=csv, file_name="predictions.csv", mime="text/csv")

            # bar chart
            summary = df['Prediction'].value_counts()
            st.bar_chart(summary)

        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")
