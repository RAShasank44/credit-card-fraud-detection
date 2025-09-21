# credit-card-fraud-detection
💳 Credit Card Fraud Detection

Project Type: Data Analyst + Machine Learning | End-to-End
This project is a Credit Card Fraud Detection system that can predict fraudulent transactions using a combination of machine learning models and anomaly detection techniques. It also includes a Streamlit dashboard for both single transaction and batch predictions.
🚀 Project Goal
Detect fraudulent credit card transactions in real-time.
Handle highly imbalanced datasets.
Provide an easy-to-use interface for manual or bulk prediction.
Explain predictions using SHAP feature importance (optional).
📂 Dataset
Source: Kaggle – Credit Card Fraud Detection(https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
Description: Contains transactions made by credit cards in September 2013 by European cardholders.
Features:
Time, Amount, V1–V28 (PCA-transformed features)
Class (target: 0 = normal, 1 = fraud)
🛠️ Technologies Used
Python Libraries: pandas, numpy, scikit-learn, imbalanced-learn, xgboost, shap, matplotlib, seaborn, joblib, streamlit
Techniques Implemented:
SMOTE: Oversampling to handle class imbalance.
Isolation Forest: Anomaly detection for fraud.
XGBoost: Gradient boosting classifier for accurate prediction.
SHAP: Feature importance explanation.
💻 How to Run Locally
Clone the repository:
git clone https://github.com/your-username/credit-card-fraud-detection.git
cd credit-card-fraud-detection
Install dependencies:
pip install -r requirements.txt
Launch Streamlit app:
streamlit run app/app.py
Open the app in your browser and start predicting.

Notes
Ensure that you downloaded the trained model xgb_fraud_model.pkl.
CSV uploads must have feature columns matching training data (Time, V1–V28, Amount).


StandardScaler: Normalization of Time and Amount.

Deployment: Streamlit Cloud
