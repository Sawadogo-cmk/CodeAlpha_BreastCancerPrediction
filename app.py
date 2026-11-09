"""
CodeAlpha - Task 4: Breast Cancer Prediction - Streamlit Demo
Interactive web app to predict breast cancer malignancy from cell nuclei features.
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Breast Cancer Risk Predictor",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ CUSTOM CSS ============
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .hero {
        background: linear-gradient(135deg, #831843 0%, #db2777 50%, #f472b6 100%);
        padding: 2.5rem 2.5rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 12px 40px rgba(131, 24, 67, 0.3);
        position: relative;
        overflow: hidden;
    }
    .hero::after {
        content: "";
        position: absolute;
        top: -60%;
        right: -5%;
        width: 500px;
        height: 500px;
        background: radial-gradient(circle, rgba(255,255,255,0.18) 0%, transparent 65%);
        border-radius: 50%;
        pointer-events: none;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(255,255,255,0.22);
        padding: 0.4rem 1rem;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 700;
        margin-bottom: 1rem;
        letter-spacing: 0.12em;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
    }
    .hero h1 {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0 0 0.6rem 0;
        letter-spacing: -0.02em;
        line-height: 1.1;
        position: relative;
        z-index: 1;
    }
    .hero p {
        font-size: 1.05rem;
        opacity: 0.95;
        margin: 0;
        font-weight: 400;
        line-height: 1.5;
        max-width: 720px;
        position: relative;
        z-index: 1;
    }

    .section-title {
        font-size: 1rem;
        font-weight: 700;
        color: #1e293b;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.6rem;
        border-bottom: 2px solid #e2e8f0;
    }

    .result-card {
        background: white;
        border-radius: 16px;
        padding: 1.75rem;
        box-shadow: 0 4px 24px rgba(15, 23, 42, 0.08);
        border: 1px solid #e2e8f0;
        margin-bottom: 1.25rem;
    }
    .result-card.high {
        border-left: 6px solid #ef4444;
        background: linear-gradient(135deg, #fef2f2 0%, #ffffff 100%);
    }
    .result-card.low {
        border-left: 6px solid #10b981;
        background: linear-gradient(135deg, #f0fdf4 0%, #ffffff 100%);
    }
    .result-card.neutral {
        background: #f8fafc;
        border-style: dashed;
        text-align: center;
    }

    .gauge-wrapper {
        display: flex;
        justify-content: center;
        margin: 1rem 0 1.5rem 0;
    }
    .gauge {
        width: 170px;
        height: 170px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        position: relative;
        box-shadow: 0 6px 24px rgba(0,0,0,0.08);
    }
    .gauge-inner {
        width: 138px;
        height: 138px;
        background: white;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2rem;
        font-weight: 800;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #500724 0%, #831843 100%);
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] li,
    [data-testid="stSidebar"] small,
    [data-testid="stSidebar"] .stMarkdown {
        color: #fbcfe8 !important;
    }
    [data-testid="stSidebar"] [data-testid="stMetricLabel"] {
        color: #f9a8d4 !important;
        font-size: 0.72rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
    }
    [data-testid="stSidebar"] [data-testid="stMetricValue"] {
        color: #f9a8d4 !important;
        font-size: 1.5rem !important;
        font-weight: 800 !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: #be185d !important;
    }

    .stButton > button, .stFormSubmitButton > button {
        background: linear-gradient(135deg, #db2777, #be185d) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.85rem 1.5rem !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        box-shadow: 0 6px 20px rgba(219, 39, 119, 0.4) !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover, .stFormSubmitButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 28px rgba(219, 39, 119, 0.5) !important;
    }

    .stSelectbox label, .stSlider label, .stNumberInput label {
        font-weight: 600 !important;
        color: #334155 !important;
        font-size: 0.88rem !important;
    }

    .footer {
        text-align: center;
        color: #64748b;
        font-size: 0.82rem;
        padding: 2.5rem 0 1rem 0;
        margin-top: 3rem;
        border-top: 1px solid #e2e8f0;
        line-height: 1.7;
    }
    .footer strong {
        color: #1e293b;
    }
</style>
""", unsafe_allow_html=True)

# ============ LOAD ARTIFACTS ============
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    features = joblib.load("features.pkl")
    medians = joblib.load("medians.pkl")
    return model, scaler, features, medians

try:
    model, scaler, feature_names, medians = load_artifacts()
except FileNotFoundError:
    st.error(
        "⚠️ Model files not found. Run the notebook's save cell to generate "
        "`model.pkl`, `scaler.pkl`, `features.pkl`, and `medians.pkl`."
    )
    st.stop()

# ============ SIDEBAR ============
with st.sidebar:
    st.markdown("## 🎗️ Breast Cancer AI")
    st.caption("Clinical decision support demo")
    st.divider()
    st.markdown("### 📊 Model Performance")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Accuracy", "98.3%")
        st.metric("F1-Score", "0.986")
    with c2:
        st.metric("ROC-AUC", "0.995")
        st.metric("Precision", "0.986")
    st.divider()
    st.markdown("### 🧠 Algorithm")
    st.caption("Logistic Regression (linear baseline won!)")
    st.divider()
    st.markdown("### 📁 Dataset")
    st.caption("""
**Breast Cancer Wisconsin**  
569 patients · 30 features  
UCI ML Repository
    """)
    st.divider()
    st.markdown("### 🔬 Top Features")
    st.caption("""
1. Worst concave points
2. Worst perimeter
3. Worst radius
4. Mean concave points
5. Worst area
    """)
    st.divider()
    st.caption("⚠️ For educational purposes only. Not a substitute for professional medical advice.")

# ============ HERO ============
st.markdown("""
<div class="hero">
    <div class="hero-badge">MACHINE LEARNING · CODEALPHA INTERNSHIP</div>
    <h1>🎗️ Breast Cancer Risk Predictor</h1>
    <p>An AI-powered clinical decision support tool trained on 569 cell nuclei samples from the UCI Breast Cancer Wisconsin dataset.</p>
</div>
""", unsafe_allow_html=True)

# ============ LAYOUT ============
left, right = st.columns([1.6, 1], gap="large")

with left:
    with st.form("prediction_form"):
        st.markdown('<div class="section-title">🔬 Top 10 Discriminative Features</div>', unsafe_allow_html=True)
        st.caption("Enter the 10 most important measurements. The 20 other features are filled with dataset medians automatically.")

        c1, c2 = st.columns(2)

        with c1:
            mean_radius = st.slider("Mean radius", 6.0, 30.0, 14.0, 0.1,
                                     help="Mean of distances from center to points on the perimeter")
            mean_perimeter = st.slider("Mean perimeter", 40.0, 200.0, 92.0, 0.5,
                                       help="Mean perimeter of the cell nuclei")
            mean_area = st.slider("Mean area", 140.0, 2500.0, 650.0, 5.0,
                                  help="Mean area of the cell nuclei")
            mean_concavity = st.slider("Mean concavity", 0.0, 0.45, 0.06, 0.005,
                                       help="Severity of concave portions of the contour")
            mean_concave_points = st.slider("Mean concave points", 0.0, 0.20, 0.03, 0.001,
                                            help="Number of concave portions of the contour")

        with c2:
            worst_radius = st.slider("Worst radius", 7.0, 40.0, 16.0, 0.1,
                                     help="Largest radius (most extreme value)")
            worst_perimeter = st.slider("Worst perimeter", 50.0, 260.0, 107.0, 0.5,
                                        help="Largest perimeter")
            worst_area = st.slider("Worst area", 185.0, 4250.0, 880.0, 5.0,
                                   help="Largest area")
            worst_concavity = st.slider("Worst concavity", 0.0, 1.3, 0.27, 0.01,
                                        help="Largest concavity value")
            worst_concave_points = st.slider("Worst concave points", 0.0, 0.30, 0.11, 0.005,
                                             help="Largest number of concave points")

        st.markdown("")
        submitted = st.form_submit_button(
            "🔍 Predict Malignancy Risk",
            type="primary",
            use_container_width=True
        )

with right:
    st.markdown('<div class="section-title">📋 Risk Assessment</div>', unsafe_allow_html=True)

    if not submitted:
        st.markdown("""
        <div class="result-card neutral">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🔬</div>
            <div style="color: #64748b; font-weight: 500; line-height: 1.6;">
                Fill in the 10 key measurements on the left and click
                <strong style="color:#1e293b;">Predict</strong> to see the malignancy risk assessment.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Start from the medians of the training data
        row = medians.copy()

        # Override with user inputs
        row["mean radius"] = mean_radius
        row["mean perimeter"] = mean_perimeter
        row["mean area"] = mean_area
        row["mean concavity"] = mean_concavity
        row["mean concave points"] = mean_concave_points
        row["worst radius"] = worst_radius
        row["worst perimeter"] = worst_perimeter
        row["worst area"] = worst_area
        row["worst concavity"] = worst_concavity
        row["worst concave points"] = worst_concave_points

        # Build DataFrame aligned with training features
        raw = pd.DataFrame([row])[feature_names]
        raw_scaled = scaler.transform(raw)

        proba = float(model.predict_proba(raw_scaled)[0, 1])  # proba of benign
        prediction = int(model.predict(raw_scaled)[0])  # 1 = benign, 0 = malignant

        # We display malignancy risk = 1 - proba(benign)
        malignancy_risk = 1 - proba

        if malignancy_risk < 0.33:
            color = "#10b981"
        elif malignancy_risk < 0.66:
            color = "#f59e0b"
        else:
            color = "#ef4444"

        st.markdown(f"""
        <div class="gauge-wrapper">
            <div class="gauge" style="background: conic-gradient({color} {malignancy_risk*100}%, #e2e8f0 {malignancy_risk*100}% 100%);">
                <div class="gauge-inner" style="color: {color};">{malignancy_risk:.0%}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if prediction == 0:  # malignant
            st.markdown("""
            <div class="result-card high">
                <div style="font-size: 2.2rem; margin-bottom: 0.5rem;">⚠️</div>
                <div style="font-size: 1.15rem; font-weight: 800; color: #dc2626; margin-bottom: 0.5rem;">
                    Malignant Tumor Detected
                </div>
                <div style="color: #475569; font-size: 0.9rem; line-height: 1.6;">
                    The model indicates a high probability that the tumor is <strong>malignant</strong>.
                    <strong>Oncology consultation and biopsy are recommended.</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:  # benign
            st.markdown("""
            <div class="result-card low">
                <div style="font-size: 2.2rem; margin-bottom: 0.5rem;">✅</div>
                <div style="font-size: 1.15rem; font-weight: 800; color: #059669; margin-bottom: 0.5rem;">
                    Benign Tumor Likely
                </div>
                <div style="color: #475569; font-size: 0.9rem; line-height: 1.6;">
                    The model indicates a high probability that the tumor is <strong>benign</strong>.
                    <strong>Routine follow-up is advised.</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="result-card" style="background: #fdf2f8;">
            <div style="font-size: 0.75rem; font-weight: 700; color: #831843; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.75rem;">
                🔬 Top Predictive Features (Random Forest)
            </div>
            <ul style="margin: 0; padding-left: 1.2rem; color: #334155; font-size: 0.88rem; line-height: 1.9;">
                <li>Worst concave points</li>
                <li>Worst perimeter</li>
                <li>Worst radius</li>
                <li>Mean concave points</li>
                <li>Worst area</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.caption("⚠️ This is a machine learning estimate, not a medical diagnosis.")

# ============ FOOTER ============
st.markdown("""
<div class="footer">
    <strong>CodeAlpha Machine Learning Internship</strong> · Built by <strong>Sawadogo Noé</strong><br>
    Breast Cancer Wisconsin dataset · Logistic Regression · Streamlit
</div>
""", unsafe_allow_html=True)