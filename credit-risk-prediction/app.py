import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ----------------------------
# Page Config
# ----------------------------

st.set_page_config(
    page_title="Credit Risk Prediction",
    page_icon="💳",
    layout="wide"
)

# ----------------------------
# Load Model & Pipeline
# ----------------------------

BASE_DIR = Path(__file__).parent

@st.cache_resource
def load_model():
    return joblib.load(BASE_DIR / "models" / "trained_model.pkl")

@st.cache_resource
def load_pipeline():
    return joblib.load(BASE_DIR / "models" / "preprocessing_pipeline.pkl")

# Load model and pipeline
model = load_model()
pipeline = load_pipeline()

# ----------------------------
# Header
# ----------------------------

st.title("💳 Credit Risk Prediction System")

st.markdown(
    """
    Predict whether a loan application is likely to be
    **Approved** or **Rejected** using Machine Learning.
    """
)

st.divider()

# ----------------------------
# Input Form
# ----------------------------

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    married = st.selectbox(
        "Married",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["0", "1", "2", "3+"]
    )

    education = st.selectbox(
        "Education",
        ["Graduate", "Not Graduate"]
    )

    self_employed = st.selectbox(
        "Self Employed",
        ["Yes", "No"]
    )

    applicant_income = st.number_input(
        "Applicant Income",
        min_value=0,
        value=5000
    )

with col2:
    coapplicant_income = st.number_input(
        "Coapplicant Income",
        min_value=0,
        value=0
    )

    loan_amount = st.number_input(
        "Loan Amount",
        min_value=0,
        value=120
    )

    loan_term = st.number_input(
        "Loan Amount Term",
        min_value=0,
        value=360
    )

    credit_history = st.selectbox(
        "Credit History",
        [1.0, 0.0]
    )

    property_area = st.selectbox(
        "Property Area",
        ["Urban", "Semiurban", "Rural"]
    )

st.divider()

# ----------------------------
# Prediction
# ----------------------------

if st.button("🔍 Predict Loan Status", use_container_width=True):

    try:
        input_df = pd.DataFrame({
            "Gender": [gender],
            "Married": [married],
            "Dependents": [dependents],
            "Education": [education],
            "Self_Employed": [self_employed],
            "ApplicantIncome": [applicant_income],
            "CoapplicantIncome": [coapplicant_income],
            "LoanAmount": [loan_amount],
            "Loan_Amount_Term": [loan_term],
            "Credit_History": [credit_history],
            "Property_Area": [property_area]
        })

        processed_input = pipeline.transform(input_df)

        prediction = model.predict(processed_input)[0]

        st.subheader("Prediction Result")

        if str(prediction).upper() == "Y":
            st.success("✅ Loan Likely to be Approved")
        else:
            st.error("❌ Loan Likely to be Rejected")

        try:
            probability = model.predict_proba(processed_input)[0]

            st.subheader("Prediction Confidence")

            confidence = float(max(probability))
            st.progress(confidence)

            st.write(
                f"Approval Probability: **{probability[1] * 100:.2f}%**"
            )

        except Exception:
            st.info("Probability score not available.")

    except Exception as e:
        st.error(f"Error: {e}")

# ----------------------------
# Footer
# ----------------------------

st.divider()

st.markdown(
    """
    **Tech Stack:** Python • Scikit-Learn • Streamlit

    Developed by **Sarthak Jadhav**
    """
)