import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# ---------------- Page Config ----------------
st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")

st.title("📊 Customer Churn Interactive Dashboard")

# ---------------- Load & Train Model ----------------
@st.cache_resource
def train_model():

    df = pd.read_csv("Telco-Customer-Churn.csv")

    # Clean data
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].mean())

    df.drop("customerID", axis=1, inplace=True)

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

# ---------------- Sidebar ----------------
st.sidebar.header("Filters")

df = full_df.copy()

# Gender Filter
gender_filter = st.sidebar.multiselect(
    "Select Gender",
    options=df["gender"].unique(),
    default=df["gender"].unique()
)
df = df[df["gender"].isin(gender_filter)]

# Contract Filter
contract_filter = st.sidebar.multiselect(
    "Select Contract",
    options=df["Contract"].unique(),
    default=df["Contract"].unique()
)
df = df[df["Contract"].isin(contract_filter)]

# ---------------- Prediction ----------------
X_pred = df.drop(["Churn"], axis=1)
X_pred = X_pred.drop("customerID", axis=1)

predictions = model.predict(X_pred)

df["Prediction"] = np.where(predictions == 1, "Churn", "No Churn")

# ---------------- KPI Section ----------------
total_customers = len(df)
churn_count = len(df[df["Prediction"] == "Churn"])
no_churn_count = len(df[df["Prediction"] == "No Churn"])
churn_rate = round((churn_count / total_customers) * 100, 2)

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
st.subheader("Detailed Customer Data")
st.dataframe(df, use_container_width=True)

# ---------------- Download ----------------
csv = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Download Predictions",
    data=csv,
    file_name="churn_predictions.csv",
    mime="text/csv"
)
