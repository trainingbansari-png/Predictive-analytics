import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("Customer Churn Prediction App")

# Load model
model = joblib.load("churn_model.pkl")

st.sidebar.header("Customer Details")

tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
monthly_charges = st.sidebar.number_input("Monthly Charges", 0.0, 200.0, 70.0)
total_charges = st.sidebar.number_input("Total Charges", 0.0, 10000.0, 1000.0)

contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

# Convert contract to numeric
contract_map = {
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2
}

input_data = np.array([[tenure, monthly_charges, total_charges, contract_map[contract]]])

if st.button("Predict"):
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ Customer is likely to Churn")
    else:
        st.success("✅ Customer is likely to Stay")
st.subheader("Dataset Overview")

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
st.dataframe(df.head())

st.subheader("Churn Distribution")
st.bar_chart(df["Churn"].value_counts())
