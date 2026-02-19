import streamlit as st
from src.pipelines.predict_pipeline import PredictPipeline

st.set_page_config(page_title="CreditIQ Engine", page_icon="💳")

st.title("CreditIQ")

# Inputs

loan_amnt = st.number_input("Loan Amount", min_value=1000.0)
annual_inc = st.number_input("Annual Income", min_value=1000.0)
fico_score = st.number_input("FICO Score", min_value=300, max_value=850)
monthly_debt = st.number_input("Monthly Debt", min_value=0.0)
term = st.selectbox("Term (Months)", [36, 60])
emp_length = st.number_input("Employment Length (Years)", min_value=0)


purpose = st.radio(
    "Purpose",
    [
        "debt_consolidation",
        "credit_card",
        "home_improvement",
        "other",
        "major_purchase",
        "small_business",
        "medical",
        "car",
        "vacation",
        "moving",
        "house",
        "wedding",
        "renewable_energy",
        "educational"
    ]
)


home_ownership = st.radio(
    "Home Ownership",
    [
        "MORTGAGE",
        "RENT",
        "OWN",
        "OTHER"
    ]
)

# Prediction

if st.button("Predict"):

    applicant = {
        "loan_amnt": float(loan_amnt),
        "annual_inc": float(annual_inc),
        "fico_score": int(fico_score),
        "monthly_debt": float(monthly_debt),
        "term": int(term),
        "emp_length": int(emp_length),
        "purpose": str(purpose),
        "home_ownership": str(home_ownership)
    }

    predictor = PredictPipeline()
    result = predictor.predict(applicant)

    st.write(result)
