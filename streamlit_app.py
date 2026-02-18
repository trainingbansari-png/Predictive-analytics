import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# ---------------- Page Config ----------------
st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")

st.title("📊 Customer Churn Interactive Dashboard")

# ---------------- Train Model ----------------
@st.cache_resource
def train_model():

    DATA_FILE = "Telco-Customer-Churn.csv"

    if not os.path.exists(DATA_FILE):
        st.error(f"{DATA_FILE} not found in repository!")
        st.stop()

    df = pd.read_csv(DATA_FILE)

    # Clean TotalCharges safely
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].mean())

    # Drop customerID safely
    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    X = df.drop("Churn", axis=1)
    y = df["Churn"].map({"No": 0, "Yes": 1})

    categorical_cols = X.select_dtypes(include="object").columns
    numeric_cols = X.select_dtypes(exclude="object").columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ],
        remainder="passthrough"
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000))
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline.fit(X_train, y_train)

    return pipeline, df


model, full_df = train_model()

# ---------------- Sidebar Filters ----------------
st.sidebar.header("Filter Customers")

df = full_df.copy()

if "gender" in df.columns:
    gender_filter = st.sidebar.multiselect(
        "Select Gender",
        options=df["gender"].unique(),
        default=df["gender"].unique()
    )
    df = df[df["gender"].isin(gender_filter)]

if "Contract" in df.columns:
    contract_filter = st.sidebar.multiselect(
        "Select Contract",
        options=df["Contract"].unique(),
        default=df["Contract"].unique()
    )
    df = df[df["Contract"].isin(contract_filter)]

# ---------------- Prediction ----------------
X_pred = df.drop("Churn", axis=1, errors="ignore")

predictions = model.predict(X_pred)

df["Prediction"] = np.where(predictions == 1, "Churn", "No Churn")

# ---------------- KPI Section ----------------
total_customers = len(df)
churn_count = len(df[df["Prediction"] == "Churn"])
no_churn_count = len(df[df["Prediction"] == "No Churn"])

churn_rate = round((churn_count / total_customers) * 100, 2) if total_customers > 0 else 0

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Customers", total_customers)
col2.metric("Churn Customers", churn_count)
col3.metric("No Churn Customers", no_churn_count)
col4.metric("Churn Rate (%)", churn_rate)

st.divider()

# ---------------- Charts ----------------
col1, col2 = st.columns(2)

with col1:
    fig1 = px.histogram(
        df,
        x="Prediction",
        title="Churn Distribution",
        text_auto=True
    )
    st.plotly_chart(fig1, use_container_width=True)

if "tenure" in df.columns:
    with col2:
        fig2 = px.box(
            df,
            x="Prediction",
            y="tenure",
            title="Tenure vs Churn"
        )
        st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ---------------- Data Table ----------------
st.subheader("Customer Data with Predictions")
st.dataframe(df, use_container_width=True)

# ---------------- Download ----------------
csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📥 Download Predictions",
    data=csv,
    file_name="churn_predictions.csv",
    mime="text/csv"
)
