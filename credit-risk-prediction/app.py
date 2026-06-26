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

model = load_model()
pipeline = load_pipeline()

# ----------------------------
# Header
# ----------------------------

st.title("💳 Credit Risk Prediction System")

st.markdown("""
Predict whether a loan application is likely to be
**Approved** or **Rejected** using Machine Learning.
""")

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
            "Loan_ID": ["LP001000"],  # Dummy Loan_ID required by pipeline
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

        # Probability Scores
        try:
            probability = model.predict_proba(processed_input)[0]

            st.subheader("Prediction Confidence")

            confidence = float(max(probability))
            st.progress(confidence)

            st.write(
                f"Approval Probability: **{probability[1] * 100:.2f}%**"
            )

            st.write(
                f"Rejection Probability: **{probability[0] * 100:.2f}%**"
            )

        except Exception:
            st.info("Probability scores not available for this model.")

    except Exception as e:
        st.error(f"Error: {e}")

# ----------------------------
# Footer
# ----------------------------

st.divider()

st.markdown("""
### 🛠 Tech Stack
- Python
- Scikit-Learn
- Pandas
- Streamlit

**Developed by Sarthak Jadhav**
""")
=======
st.set_page_config(page_title="Credit Risk Prediction", page_icon="💳", layout="wide")

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_DIR / "trained_model.pkl")
    pipeline = joblib.load(MODEL_DIR / "preprocessing_pipeline.pkl")
    return model, pipeline

model, pipeline = load_artifacts()

st.title("💳 Credit Risk Prediction")
st.write("Predict whether a loan application is likely to be approved.")

with st.form("loan_form"):
    loan_id = st.text_input("Loan ID", "LP999999")

    c1,c2 = st.columns(2)
    with c1:
        gender = st.selectbox("Gender", ["Male","Female"])
        married = st.selectbox("Married", ["Yes","No"])
        dependents = st.selectbox("Dependents", ["0","1","2","3+"])
        education = st.selectbox("Education", ["Graduate","Not Graduate"])
        self_emp = st.selectbox("Self Employed", ["No","Yes"])
        applicant_income = st.number_input("Applicant Income", min_value=0.0, value=5000.0)
    with c2:
        co_income = st.number_input("Coapplicant Income", min_value=0.0, value=0.0)
        loan_amount = st.number_input("Loan Amount", min_value=0.0, value=120.0)
        loan_term = st.selectbox("Loan Term", [12.0,36.0,60.0,84.0,120.0,180.0,240.0,300.0,360.0,480.0], index=8)
        credit_history = st.selectbox("Credit History", [1.0,0.0])
        property_area = st.selectbox("Property Area", ["Urban","Semiurban","Rural"])

    submitted = st.form_submit_button("Predict")

if submitted:
    df = pd.DataFrame([{
        "Loan_ID": loan_id,
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_emp,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": co_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "Property_Area": property_area
    }])

    try:
        X = pipeline.transform(df)
        pred = model.predict(X)[0]

        proba = None
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(X)[0]
            classes = list(model.classes_)
            if "Y" in classes:
                proba = p[classes.index("Y")]
            else:
                proba = max(p)

        st.subheader("Prediction")
        if pred == "Y":
            st.success("✅ Loan Likely Approved")
        else:
            st.error("❌ Loan Likely Rejected")

        if proba is not None:
            st.metric("Approval Probability", f"{proba*100:.2f}%")
            conf = max(proba,1-proba)
            st.metric("Confidence", f"{conf*100:.2f}%")

        st.dataframe(df)

    except Exception as e:
        st.exception(e)

