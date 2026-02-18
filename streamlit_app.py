import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ---------------- Page Config ----------------
st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")

st.title("📊 Customer Churn Interactive Dashboard")

# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# ---------------- Sidebar ----------------
st.sidebar.header("Upload & Filter Data")

uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # Clean TotalCharges
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].mean())

    # Sidebar Filters
    if "gender" in df.columns:
        gender_filter = st.sidebar.multiselect(
            "Select Gender",
            options=df["gender"].unique(),
            default=df["gender"].unique()
        )
        df = df[df["gender"].isin(gender_filter)]

    if "Contract" in df.columns:
        contract_filter = st.sidebar.multiselect(
            "Select Contract Type",
            options=df["Contract"].unique(),
            default=df["Contract"].unique()
        )
        df = df[df["Contract"].isin(contract_filter)]

    # ---------------- Predict ----------------
    if st.sidebar.button("Run Prediction"):
        try:
            predictions = model.predict(df)
            df["Prediction"] = predictions
            df["Prediction"] = df["Prediction"].map({0: "No Churn", 1: "Churn"})

            st.success("Prediction Completed ✅")

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

            # ---------------- Download Option ----------------
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Prediction Results",
                data=csv,
                file_name="churn_predictions.csv",
                mime="text/csv"
            )

            st.subheader("Detailed Data")
            st.dataframe(df, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction Error: {e}")

else:
    st.info("Please upload a CSV file from the sidebar to start.")
