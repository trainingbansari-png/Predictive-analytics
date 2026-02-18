import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, auc

# ---------------- Page Config ----------------
st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")

st.title("📊 Customer Churn Advanced Dashboard")

# ---------------- Train Model ----------------
@st.cache_resource
def train_model():

    DATA_FILE = "Telco-Customer-Churn.csv"

    if not os.path.exists(DATA_FILE):
        st.error(f"{DATA_FILE} not found!")
        st.stop()

    df = pd.read_csv(DATA_FILE)

    # Clean Data
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].mean())

    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    X = df.drop("Churn", axis=1)
    y = df["Churn"].map({"No": 0, "Yes": 1})

    categorical_cols = X.select_dtypes(include="object").columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ],
        remainder="passthrough"
    )

    model = LogisticRegression(max_iter=1000)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline.fit(X_train, y_train)

    # Accuracy
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # ROC
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    # Feature importance (Logistic Regression coefficients)
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    coefficients = pipeline.named_steps["classifier"].coef_[0]

    feature_importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": coefficients
    }).sort_values(by="Importance", key=abs, ascending=False)

    return pipeline, df, accuracy, fpr, tpr, roc_auc, feature_importance


model, full_df, accuracy, fpr, tpr, roc_auc, feature_importance = train_model()

# ---------------- Sidebar Filters ----------------
st.sidebar.header("Filters")

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
churn_rate = round((churn_count / total_customers) * 100, 2) if total_customers > 0 else 0

col1, col2, col3 = st.columns(3)
col1.metric("Total Customers", total_customers)
col2.metric("Churn Rate (%)", churn_rate)
col3.metric("Model Accuracy", round(accuracy * 100, 2))

st.divider()

# ---------------- ROC Curve ----------------
st.subheader("📈 ROC Curve")

roc_fig = go.Figure()
roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                             name=f"ROC Curve (AUC = {roc_auc:.2f})"))
roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             line=dict(dash="dash"),
                             name="Random Model"))

roc_fig.update_layout(
    xaxis_title="False Positive Rate",
    yaxis_title="True Positive Rate"
)

st.plotly_chart(roc_fig, use_container_width=True)

st.divider()

# ---------------- Feature Importance ----------------
st.subheader("🔥 Feature Importance")

top_features = feature_importance.head(15)

fig_importance = px.bar(
    top_features,
    x="Importance",
    y="Feature",
    orientation="h",
    title="Top 15 Important Features"
)

st.plotly_chart(fig_importance, use_container_width=True)

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
