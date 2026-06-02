import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import os

# ==================================================
# CONFIG
# ==================================================
st.set_page_config(
    page_title="Breast Cancer AI System",
    page_icon="🩺",
    layout="wide"
)

# ==================================================
# LOAD MODEL
# ==================================================
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "breast_cancer_rf_model.pkl")
    return joblib.load(model_path)

model = load_model()

# ==================================================
# CSS (CLEAN MEDICAL UI)
# ==================================================
st.markdown("""
<style>

.block-container {
    padding-top: 1.5rem;
}

.title {
    font-size: 40px;
    font-weight: 800;
    text-align: center;
    color: #0b3d91;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: gray;
    margin-bottom: 20px;
}

.card {
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    font-size: 22px;
    font-weight: bold;
}

.benign {
    background-color: #d4edda;
    color: #155724;
}

.malignant {
    background-color: #f8d7da;
    color: #721c24;
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# HEADER
# ==================================================
st.markdown("<div class='title'>🩺 Breast Cancer Detection System</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-Powered Clinical Decision Support Tool</div>", unsafe_allow_html=True)

st.markdown("---")

# ==================================================
# SIDEBAR
# ==================================================
with st.sidebar:
    st.header("🏥 System Info")

    st.info("""
    Model: Random Forest  
    Dataset: Breast Cancer Wisconsin  
    Type: Binary Classification  
    """)

    st.success("Model Loaded Successfully")

    st.warning("⚠️ Educational Use Only")

# ==================================================
# FEATURE LIST
# ==================================================
features = [
    "radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean",
    "compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean",

    "radius_se","texture_se","perimeter_se","area_se","smoothness_se",
    "compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se",

    "radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst",
    "compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst"
]

defaults = [17.99,10.38,122.8,1001.0,0.1184,
0.2776,0.3001,0.1471,0.2419,0.07871,
1.095,0.9053,8.589,153.4,0.006399,
0.04904,0.05373,0.01587,0.03003,0.006193,
25.38,17.33,184.6,2019.0,0.1622,
0.6656,0.7119,0.2654,0.4601,0.1189]

# ==================================================
# TABS UI
# ==================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Mean Features",
    "📈 SE Features",
    "📉 Worst Features",
    "🔍 Prediction"
])

inputs = {}

# -------------------------
# TAB 1 - MEAN FEATURES
# -------------------------
with tab1:
    st.subheader("Mean Tumor Measurements")

    for i in range(10):
        inputs[features[i]] = st.number_input(
            features[i],
            value=float(defaults[i])
        )

# -------------------------
# TAB 2 - SE FEATURES
# -------------------------
with tab2:
    st.subheader("Standard Error Features")

    for i in range(10, 20):
        inputs[features[i]] = st.number_input(
            features[i],
            value=float(defaults[i])
        )

# -------------------------
# TAB 3 - WORST FEATURES
# -------------------------
with tab3:
    st.subheader("Worst Case Features")

    for i in range(20, 30):
        inputs[features[i]] = st.number_input(
            features[i],
            value=float(defaults[i])
        )

# -------------------------
# TAB 4 - PREDICTION
# -------------------------
with tab4:

    st.subheader("Model Prediction")

    if st.button("🔬 Analyze Patient Data", use_container_width=True):

        input_df = pd.DataFrame([list(inputs.values())], columns=features)

        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]

        confidence = np.max(probabilities)

        st.markdown("---")

        # RESULT CARD
        if prediction == 0:

            st.markdown("""
            <div class='card benign'>
            ✅ BENIGN TUMOR DETECTED
            </div>
            """, unsafe_allow_html=True)

            st.balloons()

        else:

            st.markdown("""
            <div class='card malignant'>
            ⚠️ MALIGNANT TUMOR DETECTED
            </div>
            """, unsafe_allow_html=True)

        # CONFIDENCE
        st.metric("Confidence Score", f"{confidence*100:.2f}%")
        st.progress(float(confidence))

        # PROBABILITY CHART
        prob_df = pd.DataFrame({
            "Class": ["Benign", "Malignant"],
            "Probability": probabilities
        })

        fig = px.bar(
            prob_df,
            x="Class",
            y="Probability",
            text="Probability",
            color="Class",
            title="Prediction Probability"
        )

        st.plotly_chart(fig, use_container_width=True)

        # EXPANDER
        with st.expander("📖 Clinical Interpretation"):
            st.write("""
            Benign tumors are non-cancerous and usually not life-threatening.

            Malignant tumors are cancerous and may spread to other tissues.

            This tool is for educational purposes only.
            """)

# ==================================================
# FOOTER
# ==================================================
st.markdown("---")
st.caption("AI Medical Assistant • Built with Streamlit + Random Forest")