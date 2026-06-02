import streamlit as st
import pandas as pd
import joblib

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🩺",
    layout="wide"
)

# ==================================================
# CUSTOM CSS
# ==================================================
st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.title {
    text-align: center;
    font-size: 3rem;
    font-weight: bold;
    color: #FF4B4B;
}

.subtitle {
    text-align: center;
    font-size: 1.2rem;
    color: #808080;
    margin-bottom: 20px;
}

.metric-box {
    background-color: #f5f5f5;
    padding: 15px;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# LOAD MODEL
# ==================================================
@st.cache_resource
def load_model():
    return joblib.load("breast_cancer_rf_model.pkl")

model = load_model()

# ==================================================
# HEADER
# ==================================================
st.markdown(
    '<div class="title">🩺 Breast Cancer Classification System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Machine Learning Powered Early Detection Tool</div>',
    unsafe_allow_html=True
)

st.markdown("---")

# ==================================================
# SIDEBAR
# ==================================================
with st.sidebar:

    st.header("📋 Project Information")

    st.info("""
    **Model Used**
    - Random Forest Classifier

    **Dataset**
    - Breast Cancer Wisconsin Dataset

    **Purpose**
    - Predict whether a tumor is:
      - Benign
      - Malignant
    """)

    st.success("✅ Model Loaded Successfully")

    st.markdown("---")

    st.write(
        "This application is intended for educational "
        "and research purposes only."
    )

# ==================================================
# FEATURE INPUTS
# ==================================================
features = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean",
    "concave points_mean", "symmetry_mean", "fractal_dimension_mean",

    "radius_se", "texture_se", "perimeter_se", "area_se",
    "smoothness_se", "compactness_se", "concavity_se",
    "concave points_se", "symmetry_se", "fractal_dimension_se",

    "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
    "smoothness_worst", "compactness_worst", "concavity_worst",
    "concave points_worst", "symmetry_worst", "fractal_dimension_worst"
]

default_values = [
    17.99, 10.38, 122.8, 1001.0, 0.1184,
    0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
    1.095, 0.9053, 8.589, 153.4, 0.006399,
    0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
    25.38, 17.33, 184.6, 2019.0, 0.1622,
    0.6656, 0.7119, 0.2654, 0.4601, 0.1189
]

st.subheader("📊 Enter Tumor Measurements")

col1, col2 = st.columns(2)

inputs = []

for i, (feature, default) in enumerate(zip(features, default_values)):

    if i % 2 == 0:
        value = col1.number_input(
            feature,
            value=float(default),
            format="%.6f"
        )
    else:
        value = col2.number_input(
            feature,
            value=float(default),
            format="%.6f"
        )

    inputs.append(value)

# ==================================================
# PREDICT BUTTON
# ==================================================
if st.button("🔍 Predict Diagnosis", use_container_width=True):

    with st.spinner("Analyzing patient data..."):

        input_df = pd.DataFrame(
            [inputs],
            columns=features
        )

        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]

    st.markdown("---")

    st.subheader("📌 Prediction Result")

    confidence = max(probabilities)

    if prediction == 1:

        st.error("🔴 MALIGNANT TUMOR DETECTED")

        st.warning("""
        This result indicates a higher probability of malignancy.

        Please consult a qualified healthcare professional
        for proper diagnosis and medical evaluation.
        """)

    else:

        st.success("🟢 BENIGN TUMOR DETECTED")

        st.balloons()

    # ==================================================
    # CONFIDENCE SCORE
    # ==================================================
    st.subheader("🎯 Confidence Score")

    st.progress(float(confidence))

    st.metric(
        label="Model Confidence",
        value=f"{confidence * 100:.2f}%"
    )

    # ==================================================
    # PROBABILITY DISTRIBUTION
    # ==================================================
    st.subheader("📈 Probability Distribution")

    chart_df = pd.DataFrame({
        "Class": ["Benign", "Malignant"],
        "Probability": probabilities
    })

    st.bar_chart(
        chart_df.set_index("Class")
    )

    st.write(
        f"🟢 Benign Probability: {probabilities[0]*100:.2f}%"
    )

    st.write(
        f"🔴 Malignant Probability: {probabilities[1]*100:.2f}%"
    )

    # ==================================================
    # MEDICAL EXPLANATION
    # ==================================================
    with st.expander("📚 Learn More About The Prediction"):

        st.markdown("""
        ### Benign Tumor
        - Non-cancerous growth
        - Usually grows slowly
        - Less likely to spread

        ### Malignant Tumor
        - Cancerous growth
        - Can invade nearby tissues
        - Requires prompt medical attention

        ### Disclaimer
        This prediction is generated by a machine learning model
        and should not replace professional medical advice.
        """)

# ==================================================
# FOOTER
# ==================================================
st.markdown("---")

st.caption(
    "Developed using Random Forest Classifier • Streamlit • Scikit-Learn"
)