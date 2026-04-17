import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
import time
import plotly.express as px
from utils.agent import health_agent_response
from utils.predict_and_export import predict_and_export_pdf
from langchain.memory import ConversationBufferWindowMemory
st.set_page_config(
    page_title="MediRisk AI | Professional Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'history' not in st.session_state:
    st.session_state.history = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'agent_memory' not in st.session_state:
    st.session_state.agent_memory = ConversationBufferWindowMemory(k=10, memory_key="chat_history", return_messages=True)
if 'analysis_run' not in st.session_state:
    st.session_state.analysis_run = True  
if 'risk_prob' not in st.session_state:
    st.session_state.risk_prob = 0.0
if 'is_processing' not in st.session_state:
    st.session_state.is_processing = False
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "Risk Assessment"
if 'assessment_done' not in st.session_state:
    st.session_state.assessment_done = False

@st.cache_resource
def load_all_models():
    try:
        models = {
            "Logistic Regression": joblib.load('models/logistic_regression.pkl'),
            "Decision Tree": joblib.load('models/decision_tree.pkl'),
            "Random Forest": joblib.load('models/random_forest.pkl')
        }
        metrics = joblib.load('models/model_metrics.pkl')
        return models, metrics
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return {}, {}

all_models, all_metrics = load_all_models()


st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Geist:wght@100..900&family=Inter:wght@100..900&family=Outfit:wght@100..900&display=swap" rel="stylesheet">
<style>
    /* DEEP EMERALD CLINICAL DESIGN SYSTEM */
    :root {
        --primary: #0D9488;
        --primary-hover: #0F766E;
        --primary-glow: rgba(13, 148, 136, 0.15);
        --bg: #09090b;
        --sidebar: #0b0b0d;
        --surface: #141416;
        --border: rgba(255, 255, 255, 0.06);
        --text-main: #f8fafc;
        --text-dim: #94a3b8;
        --sidebar-active: rgba(13, 148, 136, 0.1);
    }

    .stApp {
        background-color: var(--bg);
        color: var(--text-main);
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3, .brand-font {
        font-family: 'Geist', sans-serif !important;
        letter-spacing: -0.04em;
    }

    [data-testid="stSidebar"] {
        background: var(--sidebar) !important;
        border-right: 1px solid var(--border) !important;
    }

    /* NAVIGATION BUTTONS */
    [data-testid="stSidebar"] button {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        color: var(--text-dim) !important;
        text-align: left !important;
        padding: 10px 16px !important;
        border-radius: 8px !important;
        margin-bottom: 8px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
    }

    [data-testid="stSidebar"] button:hover {
        background: rgba(255, 255, 255, 0.05) !important;
        border-color: rgba(16, 185, 129, 0.3) !important;
        color: #fff !important;
    }

    [data-testid="stSidebar"] button div p {
        font-size: 0.75rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }

    /* PAGE HEADER */
    .dashboard-header {
        padding: 2rem 0;
        margin-bottom: 2rem;
        border-bottom: 1px solid var(--border);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .header-title-group h1 {
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        margin: 0 !important;
    }

    /* SLEEK CLINICAL HEADERS (V4) */
    .card-heading {
    /* NATIVE CLINICAL CARDS (CLEAN V4) */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #141416 !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        padding: 2.5rem 3rem !important; /* Increased breathing room */
        margin-bottom: 2.5rem !important;
        box-shadow: 0 10px 40px rgba(0,0,0,0.5) !important;
    }

    /* LUMOS TYPOGRAPHY ENGINE (CLEAN V5) */
    .stApp {
        background-color: var(--bg);
        color: var(--text-main);
        font-family: 'Inter', sans-serif;
        font-size: 1.05rem !important; /* Global Scaling */
    }

    .clinical-label {
        font-size: 0.75rem; /* Reduced to support hierarchy */
        font-weight: 600;
        color: var(--text-dim); /* Muted to let the heading lead */
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.6rem;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #141416 !important;
        border: 1px solid var(--border) !important;
        border-radius: 16px !important;
        padding: 3.5rem !important; /* Max Breathing Room */
        margin-bottom: 3rem !important;
        box-shadow: 0 15px 50px rgba(0,0,0,0.6) !important;
    }

    /* CIRCULAR RISK HUB */
    .risk-hero {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid var(--border) !important;
        border-radius: 20px !important;
        padding: 3rem !important;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(0,0,0,0.5) !important;
        backdrop-filter: blur(10px);
    }

    .risk-score-circle {
        position: relative;
        width: 180px;
        height: 180px;
        margin-bottom: 1.5rem;
    }

    .risk-score-value {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        font-size: 3.2rem;
        font-weight: 900;
        letter-spacing: -0.05em;
        line-height: 1;
    }

    .dynamic-insight-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 15px;
        font-size: 0.9rem;
    }

    .insight-icon {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        box-shadow: 0 0 10px currentColor;
    }

    /* FORM INPUTS */
    div[data-baseweb="input"] > div, 
    div[data-baseweb="select"] > div {
        background-color: #0c0c0e !important;
        border: 1px solid var(--border) !important;
        border-radius: 6px !important;
    }

    /* SUBTLE PRIMARY CTA */
    /* PREMIUM PRIMARY CTA */
    button[kind="primary"] {
        background: rgba(16, 185, 129, 0.05) !important;
        border: 1px solid var(--primary) !important;
        color: var(--primary) !important;
        font-weight: 800 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.15em !important;
        padding: 0.8rem 2.5rem !important;
        border-radius: 12px !important;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
        height: auto !important;
        width: 100% !important;
    }

    button[kind="primary"]:hover {
        background: var(--primary) !important;
        color: #0c0c0e !important;
        box-shadow: 0 10px 30px var(--primary-glow) !important;
        transform: translateY(-2px);
    }

    /* METRIC SIMPLIFICATION */
    .metric-integrated {
        margin-bottom: 0.5rem;
    }
    .metric-track { 
        height: 1px; 
        background: rgba(255,255,255,0.08); 
        margin-top: 10px;
    }
    .metric-fill { 
        height: 100%; 
        background: var(--primary); 
        box-shadow: 0 0 12px var(--primary-glow); 
    }

    /* HIDE STREAMLIT BORDER */
    /* CLINICAL POPOVER SYNCHRONIZATION */
    div[data-baseweb="popover"] {
        z-index: 200000 !important;
        background-color: transparent !important;
    }
    
    div[data-baseweb="listbox"] {
        background-color: #0c0c0e !important;
        border: 1px solid var(--primary) !important;
        border-radius: 8px !important;
        box-shadow: 0 15px 50px rgba(0,0,0,0.9) !important;
        max-height: 280px !important;
        overflow-y: auto !important;
    }

    div[data-baseweb="listbox"] li {
        color: var(--text-main) !important;
        padding: 10px 14px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.82rem !important;
        transition: background 0.15s ease !important;
    }

    div[data-baseweb="listbox"] li:hover {
        background-color: var(--primary-glow) !important;
        color: var(--primary) !important;
    }

    /* GLOBAL CONTAINER STABILIZATION */
    section.main {
        overflow: auto !important;
    }

    #MainMenu, footer, header {visibility: hidden;}
    .stDecoration {display:none;}
</style>
""", unsafe_allow_html=True)

# SIDEBAR NAVIGATION (RESTORED PROFESSIONAL)
with st.sidebar:
    st.markdown(f"""
    <div style="margin-bottom: 3rem; padding-left: 10px;">
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="width: 32px; height: 32px; background: var(--primary); border-radius: 8px; display: flex; align-items: center; justify-content: center; box-shadow: 0 0 15px var(--primary-glow);">
                <div style="width: 16px; height: 3px; background: #000; border-radius: 2px;"></div>
                <div style="width: 3px; height: 16px; background: #000; position: absolute; border-radius: 2px;"></div>
            </div>
            <h2 class="brand-font" style="font-size: 1.2rem; font-weight: 900; margin: 0; letter-spacing: -0.05em; color: #ffffff;">
                MEDIRISK <span style="font-weight: 300; opacity: 0.4;">/ PRO</span>
            </h2>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    pages = {
        "Risk Assessment": "RISK ENGINE",
        "Patient History": "PATIENT RECORDS",
        "Analytics": "SYSTEM ANALYTICS",
        "Health Agent": "AI CONSULTANT"
    }
    
    # Custom Nav List
    selected_page = st.session_state.get('selected_page', 'Risk Assessment')
    
    for page_id, display_name in pages.items():
        # Restore Solid Active State
        if selected_page == page_id:
            st.markdown(f"""
            <style>
                div[data-testid="stSidebar"] button[key*="{page_id}"] {{
                    background: rgba(16, 185, 129, 0.1) !important;
                    border-color: rgba(16, 185, 129, 0.4) !important;
                    color: #ffffff !important;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important;
                    font-weight: 800 !important;
                }}
            </style>
            """, unsafe_allow_html=True)
            
        if st.button(display_name, key=f"nav_{page_id}", use_container_width=True):
            st.session_state.selected_page = page_id
            st.rerun()


# Update selected_page from session state
selected_page = st.session_state.get('selected_page', 'Risk Assessment')

# PAGE HEADER
st.markdown(f"""
<div class="dashboard-header">
    <div class="header-title-group">
        <h1>{selected_page}</h1>
    </div>
    <div style="display: flex; gap: 12px;">
        <div style="padding: 10px 16px; border-radius: 8px; background: rgba(255,255,255,0.02); border: 1px solid var(--border); font-size: 0.7rem; font-weight: 700; display: flex; align-items: center; gap: 8px;">
            <div style="width: 8px; height: 8px; background: #10b981; border-radius: 50%; box-shadow: 0 0 8px #10b981;"></div>
            SYSTEM: ONLINE
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
# APP LOGIC BASED ON SIDEBAR
if selected_page == "Risk Assessment":
    # Page Content Grid
    l, r = st.columns([1.6, 1], gap="large")

    with l:
        st.markdown(f"""
        <div style="display: inline-flex; align-items: center; gap: 10px; background: rgba(148, 163, 184, 0.05); border: 1px solid rgba(148, 163, 184, 0.3); padding: 8px 18px; border-radius: 6px; margin-bottom: 0.5rem; box-shadow: 0 4px 12px rgba(0,0,0,0.2);">
            <div style="width: 3px; height: 14px; background: #94a3b8; border-radius: 2px;"></div>
            <div style="font-size: 0.75rem; font-weight: 800; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.12em;">Demographics & Profile</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div style="height:15px;"></div>', unsafe_allow_html=True)
        
        with st.container(border=True):
            c1, c2 = st.columns(2, gap="large")
            with c1:
                st.markdown('<div class="clinical-label">Patient Full Name</div>', unsafe_allow_html=True)
                p_name = st.text_input("Name", placeholder="e.g. Jane Doe", label_visibility="collapsed")
                st.markdown('<div style="height:25px;"></div>', unsafe_allow_html=True)
                st.markdown('<div class="clinical-label">Biological Age</div>', unsafe_allow_html=True)
                age_val = st.number_input("Age", 1, 120, 45, label_visibility="collapsed")
            with c2:
                st.markdown('<div class="clinical-label">Sex at Birth</div>', unsafe_allow_html=True)
                sex_val = st.selectbox("Sex", ["Male", "Female"], label_visibility="collapsed")
                st.markdown('<div style="height:25px;"></div>', unsafe_allow_html=True)
                st.markdown('<div class="clinical-label">Current BMI</div>', unsafe_allow_html=True)
                bmi_val = st.number_input("BMI", 10.0, 60.0, 24.5, label_visibility="collapsed")

        st.markdown(f"""
        <div style="display: inline-flex; align-items: center; gap: 10px; background: rgba(148, 163, 184, 0.05); border: 1px solid rgba(148, 163, 184, 0.3); padding: 8px 18px; border-radius: 6px; margin-bottom: 0.5rem; box-shadow: 0 4px 12px rgba(0,0,0,0.2);">
            <div style="width: 3px; height: 14px; background: #94a3b8; border-radius: 2px;"></div>
            <div style="font-size: 0.75rem; font-weight: 800; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.12em;">Biometric Clinical Indicators</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div style="height:15px;"></div>', unsafe_allow_html=True)

        with st.container(border=True):
            
            def metric_row(label, val, unit, m_min, m_max):
                pct = (val - m_min) / (m_max - m_min) * 100
                pct = max(0, min(100, pct))
                st.markdown(f"""
                <div class="metric-integrated">
                    <div style="display:flex; justify-content:space-between; font-size:0.75rem; color:var(--text-main); font-family:'Geist Mono'; font-weight:700;">
                        <span>{label}</span>
                        <span style="color:var(--primary);">{val} {unit}</span>
                    </div>
                    <div class="metric-track" style="height:2px; margin-top:12px;">
                        <div class="metric-fill" style="width: {pct}%"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            cl1, cl2 = st.columns(2, gap="large")
            with cl1:
                st.markdown('<div class="clinical-label">Resting Blood Pressure</div>', unsafe_allow_html=True)
                bp_val = st.slider("BP", 80, 200, 120, label_visibility="collapsed")
                metric_row("GAUGE: BP", bp_val, "mmHg", 80, 200)
                st.markdown('<div style="height:35px;"></div>', unsafe_allow_html=True)
                st.markdown('<div class="clinical-label">Serum Cholesterol</div>', unsafe_allow_html=True)
                chol_val = st.slider("Chol", 100, 450, 200, label_visibility="collapsed")
                metric_row("GAUGE: CHOL", chol_val, "mg/dL", 100, 450)
            with cl2:
                st.markdown('<div class="clinical-label">Maximum Heart Rate</div>', unsafe_allow_html=True)
                hr_val = st.slider("HR", 40, 220, 150, label_visibility="collapsed")
                metric_row("GAUGE: HR", hr_val, "BPM", 40, 220)
                st.markdown('<div style="height:35px;"></div>', unsafe_allow_html=True)
                st.markdown('<div class="clinical-label">Fasting Blood Glucose</div>', unsafe_allow_html=True)
                glucose_val = st.slider("Glucose", 50, 350, 110, label_visibility="collapsed")
                metric_row("GAUGE: BS", glucose_val, "mg/dL", 50, 350)

        st.markdown(f"""
        <div style="display: inline-flex; align-items: center; gap: 10px; background: rgba(148, 163, 184, 0.05); border: 1px solid rgba(148, 163, 184, 0.3); padding: 8px 18px; border-radius: 6px; margin-bottom: 0.5rem; box-shadow: 0 4px 12px rgba(0,0,0,0.2);">
            <div style="width: 3px; height: 14px; background: #94a3b8; border-radius: 2px;"></div>
            <div style="font-size: 0.75rem; font-weight: 800; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.12em;">Clinical Diagnostic Data</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div style="height:15px;"></div>', unsafe_allow_html=True)

        with st.container(border=True):
            
            ce1, ce2 = st.columns(2, gap="large")
            with ce1:
                st.markdown('<div class="clinical-label">Chest Pain Type</div>', unsafe_allow_html=True)
                cp_val = st.selectbox("CP", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"], index=3, label_visibility="collapsed")
                st.markdown('<div style="height:25px;"></div>', unsafe_allow_html=True)
                st.markdown('<div class="clinical-label">Resting ECG Results</div>', unsafe_allow_html=True)
                ecg_val = st.selectbox("ECG", ["Normal", "ST-T abnormality", "LV hypertrophy"], label_visibility="collapsed")
            with ce2:
                st.markdown('<div class="clinical-label">Exercise Angina Observed</div>', unsafe_allow_html=True)
                exang_val = st.selectbox("Angina", ["No", "Yes"], label_visibility="collapsed")
                st.markdown('<div style="height:25px;"></div>', unsafe_allow_html=True)
                st.markdown('<div class="clinical-label">Stress Result Profile</div>', unsafe_allow_html=True)
                thal_val = st.selectbox("Stress", ["Normal", "Fixed Defect", "Reversable Defect"], label_visibility="collapsed")
            
            st.markdown('<div style="height:35px;"></div>', unsafe_allow_html=True)
            ca1, ca2, ca3 = st.columns(3, gap="large")
            with ca1:
                st.markdown('<div class="clinical-label">ST Depression</div>', unsafe_allow_html=True)
                oldpeak_val = st.slider("ST", 0.0, 6.0, 1.0, label_visibility="collapsed")
            with ca2:
                st.markdown('<div class="clinical-label">ST Slope</div>', unsafe_allow_html=True)
                slope_val = st.selectbox("Slope", ["Upsloping", "Flat", "Downsloping"], index=1, label_visibility="collapsed")
            with ca3:
                st.markdown('<div class="clinical-label">Vessel Count</div>', unsafe_allow_html=True)
                ca_val = st.slider("CA", 0, 4, 0, label_visibility="collapsed")
        st.markdown("""
        <style>
            /* GLASS-MORPHISM PRIMARY BUTTON */
            div.stButton > button[kind="primary"] {
                background: rgba(13, 148, 136, 0.15) !important;
                backdrop-filter: blur(12px) !important;
                -webkit-backdrop-filter: blur(12px) !important;
                color: #0D9488 !important;
                border: 1px solid rgba(13, 148, 136, 0.4) !important;
                box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37) !important;
                font-weight: 800 !important;
                text-transform: uppercase !important;
                letter-spacing: 0.15em !important;
                transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
            }
            div.stButton > button[kind="primary"]:hover {
                background: rgba(13, 148, 136, 0.3) !important;
                border: 1px solid rgba(13, 148, 136, 0.7) !important;
                color: #fff !important;
                box-shadow: 0 8px 32px 0 rgba(13, 148, 136, 0.2) !important;
                transform: translateY(-2px) !important;
            }
        </style>
        """, unsafe_allow_html=True)
        if st.button("RUN ASSESSMENT ENGINE", type="primary", use_container_width=True):
            missing = []

            if not p_name.strip():
                missing.append("Patient Name")

            if age_val <= 0:
                missing.append("Age")

            if bmi_val <= 0:
                missing.append("BMI")

            if missing:
                st.error(
                    "Please fill required fields: "
                    + ", ".join(missing)
                )
                st.session_state.assessment_done = False

            else:
                st.session_state.is_processing = True
                st.session_state.assessment_done = True

    # Results Section
    with r:
        if st.session_state.is_processing:
            with st.spinner("Processing clinical data..."):
                time.sleep(1.0)

        # Calculation & Results
        model = all_models.get("Random Forest")
        if model and st.session_state.assessment_done:
            st.markdown(f"""
            <div style="display: inline-flex; align-items: center; gap: 10px; background: rgba(148, 163, 184, 0.05); border: 1px solid rgba(148, 163, 184, 0.3); padding: 8px 18px; border-radius: 6px; margin-bottom: 0.5rem; box-shadow: 0 4px 12px rgba(0,0,0,0.2);">
                <div style="width: 3px; height: 14px; background: #94a3b8; border-radius: 2px;"></div>
                <div style="font-size: 0.75rem; font-weight: 800; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.12em;">Diagnostic Results Assessment</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('<div style="height:15px;"></div>', unsafe_allow_html=True)

            with st.container(border=True):
                # (Calculation logic remains same)
                s_num = 1.0 if sex_val == "Male" else 0.0
                cp_map = {"Typical Angina": 1, "Atypical Angina": 2, "Non-anginal Pain": 3, "Asymptomatic": 4}
                cp_num = cp_map.get(cp_val, 4)
                fbs_num = 1.0 if glucose_val > 120 else 0.0
                ecg_map = {"Normal": 0, "ST-T abnormality": 1, "LV hypertrophy": 2}
                ecg_num = ecg_map.get(ecg_val, 0)
                exang_num = 1.0 if exang_val == "Yes" else 0.0
                slope_map = {"Upsloping": 1, "Flat": 2, "Downsloping": 3}
                slope_num = slope_map.get(slope_val, 2)
                thal_map = {"Normal": 3, "Fixed Defect": 6, "Reversable Defect": 7}
                thal_num = thal_map.get(thal_val, 3)
                
                data = [float(age_val), s_num, cp_num, float(bp_val), float(chol_val), fbs_num, ecg_num, float(hr_val), exang_num, oldpeak_val, slope_num, float(ca_val), float(thal_num)]
                feature_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
                input_vec = pd.DataFrame([data], columns=feature_cols)
                probability = model.predict_proba(input_vec)[0][1]
                st.session_state.risk_prob = probability
                st.session_state.patient_data = {
                    'name': p_name, 'age': age_val, 'sex': s_num, 'cp': cp_num,
                    'trestbps': bp_val, 'chol': chol_val, 'fbs': fbs_num, 'restecg': ecg_num,
                    'thalach': hr_val, 'exang': exang_num, 'oldpeak': oldpeak_val, 'slope': slope_num,
                    'ca': ca_val, 'thal': thal_num, 'bmi': bmi_val, 'predicted_probability': round(probability, 4)
                }

                # PREMIUM CIRCULAR HUB
                p = probability
                if p < 0.4: lvl, color = "Low", "#10b981"
                elif p < 0.7: lvl, color = "Moderate", "#f59e0b"
                else: lvl, color = "High", "#ef4444"
                
                circumference = 2 * 3.14159 * 45
                offset = circumference * (1 - p)
                
                st.markdown(f"""
                <div class="risk-hero">
                    <div class="risk-score-circle" style="display: flex; align-items: center; justify-content: center;">
                        <svg viewBox="0 0 100 100" style="transform: rotate(-90deg); width:180px; height:180px;">
                            <circle cx="50" cy="50" r="45" fill="none" stroke="rgba(255,255,255,0.05)" stroke-width="8"/>
                            <circle cx="50" cy="50" r="45" fill="none" stroke="{color}" stroke-width="8" 
                                stroke-dasharray="{circumference}" stroke-dashoffset="{offset}" 
                                stroke-linecap="round" style="transition: stroke-dashoffset 0.8s ease;"/>
                        </svg>
                        <div class="risk-score-value" style="color:{color}; position: absolute;">{p:.2f}</div>
                    </div>
                    <div style="font-size: 0.75rem; font-weight: 800; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.15em;">
                        Clinical Risk Magnitude
                    </div>
                    <div style="margin-top:10px; font-size: 1.2rem; font-weight: 900; color: {color}; text-transform: uppercase;">
                        {lvl} Level Alert
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown('<div class="card-heading">Diagnostic Insights</div>', unsafe_allow_html=True)
                
                dynamic_points = []
                if p < 0.4: dynamic_points.append(("Overall heart trajectory looks stable.", "#10b981"))
                elif p < 0.7: dynamic_points.append(("Moderate risk profile detected. Early monitoring suggested.", "#f59e0b"))
                else: dynamic_points.append(("CRITICAL: High risk profile identified. Immediate consultation required.", "#ef4444"))

                if bp_val > 140: dynamic_points.append(("Elevated Systolic Blood Pressure detected.", "#f59e0b"))
                if chol_val > 240: dynamic_points.append(("Cholesterol levels are concerningly high.", "#ef4444"))
                if glucose_val > 120: dynamic_points.append(("High Fasting Blood Sugar detected.", "#f59e0b"))
                if bmi_val > 30: dynamic_points.append(("BMI indicates clinical obesity.", "#f59e0b"))
                if exang_val == "Yes": dynamic_points.append(("Exercise-induced angina recorded.", "#ef4444"))

                for point, p_color in dynamic_points:
                    st.markdown(f"""
                    <div class="dynamic-insight-card">
                        <div class="insight-icon" style="color: {p_color}; background: {p_color};"></div>
                        <div>{point}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown('<div style="height:35px;"></div>', unsafe_allow_html=True)
                st.markdown('<div class="card-heading">Dynamic Key Drivers</div>', unsafe_allow_html=True)
                
                inner = model.named_steps['model']
                features = ['Age', 'Sex', 'CP', 'BP', 'Chol', 'Fasting Sugar', 'ECG', 'Heart Rate', 'Exercise Angina', 'Peak', 'Slope', 'CA', 'Thal']
                
                if hasattr(inner, 'feature_importances_'):
                    global_imps = inner.feature_importances_
                elif hasattr(inner, 'coef_'):
                    global_imps = np.abs(inner.coef_[0])
                    global_imps = global_imps / np.sum(global_imps)
                else:
                    global_imps = [0.1] * 13
                    
                local_influences = []
                for i, feat in enumerate(features):
                    base = global_imps[i]
                    multiplier = 1.0
                    if feat == 'BP': multiplier += abs(bp_val - 120) / 60.0
                    elif feat == 'Chol': multiplier += abs(chol_val - 200) / 100.0
                    elif feat == 'Fasting Sugar' and glucose_val > 120: multiplier += 1.5
                    elif feat == 'Age': multiplier += abs(age_val - 45) / 40.0
                    elif feat == 'Heart Rate': multiplier += abs(hr_val - 150) / 60.0
                    elif feat == 'CA': multiplier += (ca_val * 0.4)
                    elif feat == 'Peak': multiplier += (oldpeak_val * 0.3)
                    
                    local_influences.append(base * multiplier)
                    
                total_inf = sum(local_influences)
                local_influences = [v/total_inf for v in local_influences]
                
                feature_imp = dict(sorted(zip(features, local_influences), key=lambda x: x[1], reverse=True)[:5])
                
                for feat, val in feature_imp.items():
                    st.markdown(f"""
                    <div style="display:flex; justify-content:space-between; margin-bottom:10px; font-size:0.75rem;">
                        <span style="color:#94a3b8; font-weight:700;">{feat}</span>
                        <span style="color:#a855f7; font-weight:900;">{int(val*100)}% Influence</span>
                    </div>
                    <div style="height:4px; background:rgba(255,255,255,0.03); border-radius:10px; margin-bottom:20px; overflow:hidden; border: 1px solid rgba(255,255,255,0.05);">
                        <div style="height:100%; width:{int(val*100)}%; background: linear-gradient(90deg, #94a3b8, #a855f7); border-radius:10px; box-shadow: 0 0 15px rgba(168, 85, 247, 0.3);"></div>
                    </div>
                    """, unsafe_allow_html=True)

                st.divider()
                
                cl1, cl2 = st.columns(2)
                with cl1:
                    if st.button("Save Assessment", use_container_width=True):
                        if not p_name or p_name.strip() == "":
                            st.error("Patient Name required.")
                        else:
                            st.session_state.history.append({
                                "Name": p_name,
                                "Risk Score": round(p, 2),
                                "Risk Level": lvl,
                                "Date": time.strftime("%Y-%m-%d %H:%M")
                            })
                            st.success("Saved.")
                with cl2:
                    if st.button("Export PDF", use_container_width=True):
                        if not p_name or p_name.strip() == "":
                            st.error("Patient Name required.")
                        else:
                            try:
                                pdf_path = predict_and_export_pdf(
                                    {k: v for k, v in st.session_state.patient_data.items() if k in feature_cols},
                                    patient_name=p_name,
                                    file_name=f"report_{int(time.time())}"
                                )
                                with open(pdf_path, "rb") as f:
                                    pdf_bytes = f.read()
                                st.download_button("Download File", data=pdf_bytes, file_name=pdf_path.name, mime="application/pdf")
                            except Exception as e:
                                st.error(f"Error generating PDF: {e}")

            # Minimalist CTA
            st.markdown(f"""
            <div style="background: rgba(16, 185, 129, 0.05); padding: 20px; border: 1px solid rgba(16, 185, 129, 0.1); border-radius: 8px; margin-top: 24px;">
                <div style="color: var(--primary); font-size: 0.75rem; font-weight: 700; text-transform: uppercase; margin-bottom: 8px;">Next Phase Recommendation</div>
                <div style="color: #f8fafc; font-size: 0.9rem; line-height: 1.5;">
                    Visit the <b>Health Agent</b> via the sidebar to discuss these results with our clinical AI assistant.
                </div>
            </div>
            """, unsafe_allow_html=True)

            # stop loading after rendering results
            st.session_state.is_processing = False
        else:
            # Placeholder or info when no assessment is run
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.02); border: 1px dashed var(--border); border-radius: 12px; padding: 3rem; text-align: center; margin-top: 1rem;">
                <div style="font-size: 0.7rem; color: var(--primary); font-weight: 800; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 1rem;">Clinical Intelligence Engine</div>
                <div style="color: var(--text-dim); font-size: 0.85rem; line-height: 1.6;">
                    Complete the patient health profile on the left <br> and engage the risk engine to generate diagnostic insights.
                </div>
            </div>
            """, unsafe_allow_html=True)

# HISTORY & ANALYTICS WRAPPER
elif selected_page in ["Patient History", "Analytics"]:
    if selected_page == "Patient History":
        st.markdown('<div class="card-heading">Clinical Records Database</div>', unsafe_allow_html=True)
        if len(st.session_state.history) == 0:
            st.info("No records found in current session database.")
        else:
            df = pd.DataFrame(st.session_state.history)
            st.dataframe(df, use_container_width=True)
    else:
        st.markdown('<div class="card-heading">Population Analytics Dashboard</div>', unsafe_allow_html=True)
        if len(st.session_state.history) == 0:
            st.info("Insufficient data for clinical analytics. Please perform assessments first.")
        else:
            df = pd.DataFrame(st.session_state.history)
            avg_risk = df["Risk Score"].mean()
            
            # Clinical Status Logic
            if avg_risk < 0.35:
                status_label, status_color = "NOMINAL (STABLE)", "#10b981"
            elif avg_risk < 0.65:
                status_label, status_color = "MODERATE (MONITOR)", "#f59e0b"
            else:
                status_label, status_color = "CRITICAL (ELEVATED)", "#ef4444"
                
            # Status Badge Summary
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.02); border: 1px solid var(--border); border-radius: 12px; padding: 1.5rem; margin-bottom: 2rem; display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <div style="font-size: 0.65rem; color: var(--text-dim); font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 4px;">Current Population Health Status</div>
                    <div style="font-size: 1.8rem; font-weight: 900; color: {status_color}; letter-spacing: -0.02em;">{status_label}</div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 0.65rem; color: var(--text-dim); font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 4px;">Mean Risk Magnitude</div>
                    <div style="font-size: 1.8rem; font-weight: 900; color: #fff;">{avg_risk:.2f}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            ca1, ca2 = st.columns(2, gap="large")
            with ca1:
                st.markdown('<div class="card-heading">Risk Distribution</div>', unsafe_allow_html=True)
                counts = df["Risk Level"].value_counts().reset_index()
                counts.columns = ["Risk Level", "Count"]
                
                level_order = ["Low", "Moderate", "High"]
               
                glass_color_map = {
                    "Low": "rgba(45, 212, 191, 0.4)",     
                    "Moderate": "rgba(245, 158, 11, 0.4)", 
                    "High": "rgba(239, 68, 68, 0.4)"      
                }
                
                fig_dist = px.bar(
                    counts, x="Risk Level", y="Count",
                    color="Risk Level",
                    color_discrete_map=glass_color_map,
                    category_orders={"Risk Level": level_order},
                    template="plotly_dark",
                    text="Count"
                )
                fig_dist.update_traces(
                    marker_line_color='rgba(255,255,255,0.4)',
                    marker_line_width=1.5,
                    textposition='outside',
                    textfont_color='rgba(255,255,255,0.7)',
                    cliponaxis=False
                )
                fig_dist.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    showlegend=False,
                    margin=dict(t=35, b=20, l=20, r=20),
                    font=dict(family="Inter", size=10),
                    xaxis=dict(title=None, showgrid=False),
                    yaxis=dict(title=None, showgrid=True, gridcolor="rgba(255,255,255,0.03)")
                )
                st.plotly_chart(fig_dist, use_container_width=True, config={"displayModeBar": False})
                
                # SEAMLESS DIAGNOSTIC FOOTER
                if not counts.empty:
                    majority_level = counts.sort_values("Count", ascending=False).iloc[0]["Risk Level"]
                    percent = (counts.sort_values("Count", ascending=False).iloc[0]["Count"] / counts["Count"].sum()) * 100
                    st.markdown(f"""
                    <div style="border-top: 1px solid rgba(255,255,255,0.05); padding-top: 12px; margin-top: 5px; display: flex; align-items: center; gap: 8px;">
                        <div style="width: 6px; height: 6px; border-radius: 50%; background: #0D9488;"></div>
                        <div style="font-size: 0.72rem; color: #94a3b8; font-family: 'Inter';">CONCLUSION: Population density established in <span style="color:#0D9488; font-weight:700;">{majority_level} Risk</span> ({percent:.1f}%)</div>
                    </div>
                    """, unsafe_allow_html=True)

            with ca2:
                st.markdown('<div class="card-heading">Temporal Risk Trends</div>', unsafe_allow_html=True)
                df["Date"] = pd.to_datetime(df["Date"])
                df_sorted = df.sort_values("Date")
                
                fig_trend = px.area(
                    df_sorted, x="Date", y="Risk Score",
                    template="plotly_dark"
                )
                fig_trend.update_traces(
                    line_color="#71938F",
                    line_width=3,
                    fillcolor="rgba(113, 147, 143, 0.1)",
                    marker=dict(size=8, color="#71938F", line=dict(width=2, color="rgba(255,255,255,0.6)"))
                )
                fig_trend.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(t=20, b=20, l=20, r=20),
                    font=dict(family="Inter", size=10),
                    xaxis=dict(title=None, showgrid=False),
                    yaxis=dict(title=None, showgrid=True, gridcolor="rgba(255,255,255,0.03)")
                )
                
                st.plotly_chart(fig_trend, use_container_width=True, config={"displayModeBar": False})
                
                # SEAMLESS TREND ANALYTICS
                if len(df_sorted) > 1:
                    delta = df_sorted.iloc[-1]["Risk Score"] - df_sorted.iloc[0]["Risk Score"]
                    trend = "STABLE/DESCENDING" if delta <= 0 else "ELEVATING"
                    t_color = "#10b981" if delta <= 0 else "#ef4444"
                    st.markdown(f"""
                    <div style="border-top: 1px solid rgba(255,255,255,0.05); padding-top: 12px; margin-top: 5px; display: flex; align-items: center; gap: 8px;">
                        <div style="width: 6px; height: 6px; border-radius: 50%; background: {t_color};"></div>
                        <div style="font-size: 0.72rem; color: #94a3b8; font-family: 'Inter';">CONCLUSION: Analytical trajectory is <span style="color:{t_color}; font-weight:700;">{trend}</span> ({delta:+.2f} magnitude)</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="border-top: 1px solid rgba(255,255,255,0.05); padding-top: 12px; margin-top: 5px; display: flex; align-items: center; gap: 8px;">
                        <div style="width: 6px; height: 6px; border-radius: 50%; background: #94a3b8;"></div>
                        <div style="font-size: 0.72rem; color: #94a3b8; font-family: 'Inter';">CONCLUSION: <span style="color:#94a3b8; font-weight:700;">INSUFFICIENT DATA</span> for temporal trend analysis.</div>
                    </div>
                    """, unsafe_allow_html=True)

elif selected_page == "Health Agent":
    # Custom Chat UI Styles
    st.markdown("""
    <style>
        .stChatMessage {
            background: rgba(255, 255, 255, 0.01) !important;
            border: 1px solid var(--border-glass) !important;
            border-radius: 20px !important;
            padding: 1rem !important;
            margin-bottom: 1rem !important;
            backdrop-filter: blur(5px);
        }
        .stChatMessage [data-testid="stChatMessageAvatarUser"] {
            background-color: var(--primary) !important;
        }
    </style>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([1, 2.8], gap="large")
    
    with c1:
        st.markdown('<div class="card-heading">Data Source Selection</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload Patient Health Doc (.pdf)", type=["pdf"], label_visibility="collapsed")
        if uploaded_file and st.button("INDEX", use_container_width=True):
            from utils.rag import process_document
            with st.spinner("Analyzing doc..."):
                st.session_state.vectorstore = process_document(uploaded_file)
                st.success("Context loaded.")
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("RESET MEMORY", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.agent_memory.clear()
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card-heading">Clinical AI Consultant</div>', unsafe_allow_html=True)
        
        # Spacer to push input to bottom if needed, but Streamlit handles chat_input
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(f'<div style="font-size:0.9rem; line-height:1.6; font-family:\'Inter\';">{msg["content"]}</div>', unsafe_allow_html=True)
        
        user_input = st.chat_input("Query clinical indicators...")
        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            st.chat_message("user").markdown(user_input)
            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    answer, tools = health_agent_response(
                        user_input, st.session_state.get('patient_data', {}),
                        st.session_state.risk_prob, st.session_state.get('vectorstore'),
                        st.session_state.agent_memory
                    )
                    st.markdown(f'<div style="font-size:0.9rem; line-height:1.6; font-family:\'Inter\';">{answer}</div>', unsafe_allow_html=True)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.markdown('</div>', unsafe_allow_html=True)


st.markdown('<div style="text-align: center; color: var(--text-color); opacity: 0.5; font-size: 0.7rem; margin-top: 40px; padding: 20px;">VERIFIED FOR CLINICAL EVALUATION • MEDIRISK PRO V2.5</div>', unsafe_allow_html=True)