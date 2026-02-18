import streamlit as st
import pandas as pd
import joblib

# Load the pre-trained model and scaler
model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')

# Create a Streamlit app
st.title("Customer Churn Prediction")

# Input features
tenure = st.slider("Tenure", 0, 72, 10)
monthly_charges = st.number_input("Monthly Charges", min_value=0, max_value=1000, value=70)
total_charges = st.number_input("Total Charges", min_value=0, max_value=100000, value=3000)

# Prepare input features for prediction
features = pd.DataFrame([[tenure, monthly_charges, total_charges]], columns=['tenure', 'MonthlyCharges', 'TotalCharges'])

# Scale the features using the same scaler that was used during training
features_scaled = scaler.transform(features)

# Make prediction
prediction = model.predict(features_scaled)

# Display the result
if prediction[0] == 1:
    st.write("The customer will churn.")
else:
    st.write("The customer will not churn.")
