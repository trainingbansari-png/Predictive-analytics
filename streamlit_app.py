import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Load pre-trained model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Create a Streamlit app
st.title("Customer Churn Prediction")

# Input features
tenure = st.slider("Tenure", 0, 72, 10)
monthly_charges = st.number_input("Monthly Charges", min_value=0, max_value=1000, value=70)
total_charges = st.number_input("Total Charges", min_value=0, max_value=100000, value=3000)

# Make prediction
features = pd.DataFrame([[tenure, monthly_charges, total_charges]], columns=['tenure', 'MonthlyCharges', 'TotalCharges'])
prediction = model.predict(features)

# Display the result
if prediction == 1:
    st.write("The customer will churn.")
else:
    st.write("The customer will not churn.")
