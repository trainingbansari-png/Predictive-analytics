# streamlit_app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# Load the trained model and scaler
model = joblib.load('models/churn_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Load data
df = pd.read_csv('data/telco-customer-churn.csv')

# Set page title
st.title("Customer Churn Prediction Dashboard")

# Sidebar filters
st.sidebar.header("Filter Customers")
tenure = st.sidebar.slider("Select Customer Tenure", 0, 72, (0, 72))
churn_status = st.sidebar.selectbox("Select Churn Status", ['All', 'Churned', 'Not Churned'])

# Filter data based on selected filters
filtered_df = df[df['tenure'].between(tenure[0], tenure[1])]
if churn_status != 'All':
    filtered_df = filtered_df[filtered_df['Churn'] == (1 if churn_status == 'Churned' else 0)]

# Show filtered data in the main panel
st.write("### Filtered Customer Data", filtered_df)

# Display a churn count bar chart
churn_count = df['Churn'].value_counts()
st.write("### Churn Count")
fig = px.bar(churn_count, x=churn_count.index, y=churn_count.values, labels={'index': 'Churn Status', 'y': 'Count'})
st.plotly_chart(fig)

# Scatter plot for Tenure vs Monthly Charges
fig = px.scatter(df, x="tenure", y="MonthlyCharges", color="Churn", title="Tenure vs Monthly Charges")
st.plotly_chart(fig)

# User input for prediction
st.sidebar.header("Predict Customer Churn")
tenure_input = st.sidebar.slider("Tenure", 0, 72, 10)
monthly_charges_input = st.sidebar.number_input("Monthly Charges", min_value=0, max_value=1000, value=70)
total_charges_input = st.sidebar.number_input("Total Charges", min_value=0, max_value=100000, value=3000)
gender_input = st.sidebar.selectbox("Gender", ["Male", "Female"])

# Convert gender to numeric
gender_numeric = 1 if gender_input == "Female" else 0

# Prepare input for prediction
features = pd.DataFrame([[tenure_input, monthly_charges_input, total_charges_input, gender_numeric]], 
                        columns=['tenure', 'MonthlyCharges', 'TotalCharges', 'Gender'])

# Scale the input features using the loaded scaler
features_scaled = scaler.transform(features)

# Make prediction
prediction = model.predict(features_scaled)

# Display the result
if prediction == 1:
    st.sidebar.write("The customer is predicted to churn.")
else:
    st.sidebar.write("The customer is predicted to not churn.")
