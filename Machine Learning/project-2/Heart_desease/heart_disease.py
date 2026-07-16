import os
import joblib
import pandas as pd
import streamlit as st

# ---- 1. PREMIUM DARK CONFIGURATION ----
st.set_page_config(
    page_title="CardioCare | Dark Premium Dashboard",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---- 2. HUMAN-DESIGNED DARK CORPORATE CSS ----
st.markdown("""
    <style>
    /* Main body background - Premium Deep Slate Dark */
    .stApp {
        background-color: #0f172a;
    }
    
    /* Clean Minimalist Dark Header */
    .app-header {
        text-align: left;
        padding: 20px 0px 10px 0px;
        margin-bottom: 25px;
        border-bottom: 1px solid #334155;
    }
    .main-title {
        color: #ffffff;
        font-size: 32px;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin: 0;
    }
    .subtitle {
        color: #94a3b8;
        font-size: 15px;
        margin-top: 6px;
    }

    /* Human Created Structured Dark Cards */
    .premium-dark-card {
        background: #1e293b;
        padding: 25px;
        border-radius: 12px;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 25px;
    }
    .card-title {
        font-size: 16px;
        font-weight: 600;
        color: #cbd5e1;
        margin-bottom: 20px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Solid Matte Sky Blue CTA Button - High contrast, pure professional */
    div.stButton > button:first-child {
        background-color: #38bdf8;
        color: #0f172a !important;
        font-weight: 700;
        font-size: 16px;
        padding: 12px 24px;
        border-radius: 8px;
        border: none;
        width: 100%;
        cursor: pointer;
        transition: background-color 0.2s ease, transform 0.1s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    div.stButton > button:first-child:hover {
        background-color: #0ea5e9;
        transform: translateY(-1px);
    }
    div.stButton > button:first-child:active {
        transform: translateY(0);
    }
    
    /* Ensuring all input text labels remain clearly visible on dark theme */
    label {
        color: #f1f5f9 !important;
    }
    </style>
""", unsafe_allow_html=True)

# ---- 3. ML MODEL ENGINE LOADING ----
model = joblib.load("knn_heart_model.pkl")
scaler = joblib.load("heart_scaler.pkl")

# ---- 4. CLEAN CLINICAL DARK HEADER ----
st.markdown("""
    <div class="app-header">
        <div class="main-title">❤️ Cardiovascular Risk Analytics</div>
        <div class="subtitle">Clinical decision support tool • Prepared by Hardik Sachan</div>
    </div>
""", unsafe_allow_html=True)

# ---- 5. TWO-COLUMN RESPONSIVE LAYOUT GRID ----
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="premium-dark-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">👤 Patient Demographics</div>', unsafe_allow_html=True)
    age = st.slider("Age (Years)", 18, 100, 40)
    sex = st.selectbox("Biological Sex", ["M", "F"])
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
    restingBP = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    cholestrol = st.number_input("Serum Cholesterol (mg/dL)", 100, 600, 200)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="premium-dark-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🩺 Clinical Observations</div>', unsafe_allow_html=True)
    fastingBS = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
    restingECG = st.selectbox("Resting ECG Results", ["Normal", "ST", "LVH"])
    max_hr = st.slider("Maximum Heart Rate (bpm)", 60, 220, 150)
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
    oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0)
    st_slope = st.selectbox("Peak Exercise ST Slope", ["Up", "Flat", "Down"])
    st.markdown('</div>', unsafe_allow_html=True)

# ---- 6. COMPUTE ENGINE & MATTE RESULT DISPLAY ----
if st.button("Analyze Patient Data"):
    raw_input = {
        "Age": age,
        "RestingBP": restingBP,
        "Cholesterol": cholestrol,
        "FastingBS": fastingBS,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        "Sex_" + sex: 1,
        "ChestPainType_" + chest_pain: 1,
        "RestingECG_" + restingECG: 1,
        "ExerciseAngina_" + exercise_angina: 1,
        "ST_Slope_" + st_slope: 1,
    }

    input_df = pd.DataFrame([raw_input])
    input_df = input_df.reindex(columns=scaler.feature_names_in_, fill_value=0)
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)

    st.write("")
    st.markdown("### 📊 Diagnostic Evaluation Summary")
    
    if prediction == 1:
        st.error("""
        **High Risk Profile Detected**
        
        The clinical analysis indicates statistical parameters aligned with high-risk cardiovascular patterns. 
        
        *Recommendation: Refer the patient for a complete cardiology screening and secondary lab assessment.*
        """)
    else:
        st.success("""
        **Low Risk Profile Detected**
        
        The analysis indicates that the patient's parameters currently track within acceptable cardiovascular baselines.
        
        *Recommendation: Continue routine screening panel controls and normal physiological health guidelines.*
        """)
