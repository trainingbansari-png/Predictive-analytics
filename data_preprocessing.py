# data_preprocessing.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load data
df = pd.read_csv('data/telco-customer-churn.csv')

# Data Preprocessing
# Fill missing values (if any)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].mean(), inplace=True)

# Convert categorical columns to numeric (Churn, gender, etc.)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
df['Gender'] = df['Gender'].map({'Female': 1, 'Male': 0})

# Select relevant features
X = df[['tenure', 'MonthlyCharges', 'TotalCharges', 'Gender']]
y = df['Churn']

# Normalize features using MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler for later use
import joblib
joblib.dump(scaler, 'models/scaler.pkl')

# Return processed data
X_scaled, y
