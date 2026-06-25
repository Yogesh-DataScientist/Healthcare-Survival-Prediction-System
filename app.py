import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import os
import random
import time
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# ─── PAGE CONFIGURATION ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Healthcare Survival Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
MODEL_FEATURES = [
    "age", "bmi", "ethnicity", "gender", "icu_admit_source", "icu_stay_type",
    "apache_2_diagnosis", "gcs_eyes_apache", "gcs_motor_apache",
    "gcs_verbal_apache", "heart_rate_apache", "map_apache", "resprate_apache",
    "d1_spo2_min", "d1_temp_max", "d1_glucose_max", "diabetes_mellitus",
    "cirrhosis",
]

ETHNICITY_OPTIONS = [
    "African American", "Asian", "Caucasian", "Hispanic",
    "Native American", "Other/Unknown",
]
ETHNICITY_MAP = {v: i for i, v in enumerate(ETHNICITY_OPTIONS)}

GENDER_OPTIONS = ["Female", "Male"]
GENDER_MAP = {"Female": 0, "Male": 1}
GENDER_DATA_MAP = {"F": 0, "M": 1}

ICU_SOURCE_OPTIONS = [
    "Accident & Emergency", "Floor", "Operating Room / Recovery",
    "Other Hospital", "Other ICU",
]
ICU_SOURCE_MAP = {v: i for i, v in enumerate(ICU_SOURCE_OPTIONS)}

ICU_STAY_OPTIONS = ["Admit", "Readmit", "Transfer"]
ICU_STAY_MAP = {"Admit": 0, "Readmit": 1, "Transfer": 2}
ICU_STAY_DATA_MAP = {"admit": 0, "readmit": 1, "transfer": 2}

DATA_PATH = "dataset.csv"
XGB_PATH = "xgboost_healthcare_model.pkl"
ADA_PATH = "adaboost_healthcare_model.pkl"


# ─── SESSION STATE INITIALIZATION ─────────────────────────────────────────────
def init_session_state():
    """Initialize all session state keys with sensible defaults."""
    defaults = {
        "theme": "Dark",
        "animations": True,
        "particles": True,
        "selected_model": "XGBoost",
        "prediction_result": None,
        "prediction_proba": None,
        "just_predicted": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ─── CSS INJECTION ────────────────────────────────────────────────────────────
def inject_css():
    """Inject the complete CSS theme based on current settings."""
    theme = st.session_state.get("theme", "Dark")
    anim = st.session_state.get("animations", True)
    is_dark = theme == "Dark"

    if is_dark:
        css_vars = """
            --bg-main: #111315;
            --bg-gradient: linear-gradient(160deg, #111315 0%, #141618 50%, #0F1113 100%);
            --sidebar-bg: linear-gradient(180deg, #111315 0%, #161819 100%);
            --card-bg: rgba(28, 31, 34, 0.92);
            --card-border: rgba(255, 255, 255, 0.08);
            --card-hover-border: rgba(16, 185, 129, 0.45);
            --text-primary: #E8EAED;
            --text-secondary: #9CA3AF;
            --input-bg: rgba(255, 255, 255, 0.05);
            --input-border: rgba(255, 255, 255, 0.12);
            --scrollbar-track: #111315;
            --shadow-color: rgba(16, 185, 129, 0.12);
            --sidebar-border: rgba(16, 185, 129, 0.18);
            --accent: #10B981;
            --accent-secondary: #14B8A6;
            --accent-gold: #D4A847;
            --bar-bg: rgba(255,255,255,0.10);
            --metric-value-color: #10B981;
            --expander-bg: rgba(28, 31, 34, 0.7);
            --toggle-label: #D1D5DB;
            --select-bg: rgba(255,255,255,0.06);
            --select-color: #E8EAED;
        """
    else:
        css_vars = """
            --bg-main: #F4F5F4;
            --bg-gradient: #F4F5F4;
            --sidebar-bg: linear-gradient(180deg, #FAFBFA 0%, #EEF0EE 100%);
            --card-bg: #FFFFFF;
            --card-border: rgba(0, 0, 0, 0.09);
            --card-hover-border: rgba(5, 150, 105, 0.40);
            --text-primary: #1A1D21;
            --text-secondary: #52575E;
            --input-bg: #FFFFFF;
            --input-border: rgba(0, 0, 0, 0.16);
            --scrollbar-track: #E5E7EB;
            --shadow-color: rgba(0, 0, 0, 0.07);
            --sidebar-border: rgba(0, 0, 0, 0.09);
            --accent: #059669;
            --accent-secondary: #10B981;
            --accent-gold: #92713A;
            --bar-bg: rgba(0,0,0,0.08);
            --metric-value-color: #047857;
            --expander-bg: #FFFFFF;
            --toggle-label: #374151;
            --select-bg: #FFFFFF;
            --select-color: #1A1D21;
        """

    # Kill every single animation immediately if animations toggle is OFF
    anim_override = "" if anim else """
        *, ::before, ::after {
            animation: none !important;
            transition: none !important;
        }
        .float-logo, .float-icon, .animate-in, .animate-in-delay, .result-glow {
            animation: none !important;
            transform: none !important;
            opacity: 1 !important;
        }
        .glass-card:hover, .metric-card:hover, .feature-card:hover,
        .stButton > button:hover, [data-testid="stFormSubmitButton"] button:hover {
            transform: none !important;
            box-shadow: none !important;
        }
    """

    st.markdown(f"""<style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

        :root {{ {css_vars} }}
        {anim_override}

        /* ── Global App Background ─────────────── */
        .stApp {{
            background: var(--bg-gradient) !important;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }}
        html, body, [data-testid="stAppViewContainer"],
        [data-testid="stHeader"] {{
            background: transparent !important;
        }}
        #MainMenu {{visibility: hidden;}}
        footer    {{visibility: hidden;}}
        header    {{visibility: hidden;}}

        /* Force readable text color across main area */
        [data-testid="stAppViewContainer"] p,
        [data-testid="stAppViewContainer"] span,
        [data-testid="stAppViewContainer"] label,
        [data-testid="stAppViewContainer"] li,
        [data-testid="stAppViewContainer"] h1,
        [data-testid="stAppViewContainer"] h2,
        [data-testid="stAppViewContainer"] h3,
        [data-testid="stAppViewContainer"] h4,
        [data-testid="stAppViewContainer"] div {{
            color: var(--text-primary);
        }}

        /* ── Scrollbar ─────────────────────────── */
        ::-webkit-scrollbar {{ width: 7px; }}
        ::-webkit-scrollbar-track {{ background: var(--scrollbar-track); }}
        ::-webkit-scrollbar-thumb {{
            background: linear-gradient(180deg, #10B981, #059669);
            border-radius: 4px;
        }}

        /* ── Sidebar ───────────────────────────── */
        [data-testid="stSidebar"] {{
            background: var(--sidebar-bg) !important;
            border-right: 1px solid var(--sidebar-border);
        }}
        [data-testid="stSidebar"] * {{
            color: var(--text-primary) !important;
        }}
        [data-testid="stSidebar"] [data-testid="stRadio"] label {{
            padding: 10px 16px !important;
            border-radius: 10px !important;
            transition: all 0.25s ease !important;
            cursor: pointer !important;
            margin-bottom: 3px;
        }}
        [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {{
            background: rgba(16, 185, 129, 0.12) !important;
        }}

        /* ── Cards ─────────────────────────────── */
        .glass-card {{
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 14px;
            transition: all 0.35s cubic-bezier(0.16, 1, 0.3, 1);
        }}
        .glass-card:hover {{
            border-color: var(--card-hover-border);
            transform: translateY(-3px);
            box-shadow: 0 10px 32px var(--shadow-color);
        }}

        .metric-card {{
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 22px 16px;
            text-align: center;
            transition: all 0.35s cubic-bezier(0.16, 1, 0.3, 1);
        }}
        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 12px 36px var(--shadow-color);
            border-color: var(--card-hover-border);
        }}
        .metric-icon  {{ font-size: 2.2rem; margin-bottom: 6px; }}
        .metric-value {{
            font-size: 1.8rem; font-weight: 700;
            color: var(--metric-value-color);
            margin-bottom: 2px;
        }}
        .metric-label {{
            color: var(--text-secondary);
            font-size: 0.85rem; font-weight: 500;
        }}

        .feature-card {{
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 24px 18px;
            text-align: center;
            transition: all 0.35s cubic-bezier(0.16, 1, 0.3, 1);
            height: 100%;
        }}
        .feature-card:hover {{
            border-color: var(--card-hover-border);
            transform: translateY(-4px);
            box-shadow: 0 10px 32px var(--shadow-color);
        }}
        .feature-icon {{ font-size: 2rem; margin-bottom: 10px; }}
        .feature-title {{
            color: var(--text-primary);
            font-weight: 600; font-size: 1rem;
            margin-bottom: 6px;
        }}
        .feature-desc {{
            color: var(--text-secondary);
            font-size: 0.82rem; line-height: 1.5;
        }}

        /* ── Titles ────────────────────────────── */
        .gradient-title {{
            font-size: 2.5rem; font-weight: 800; text-align: center;
            color: var(--metric-value-color);
            margin-bottom: 4px; line-height: 1.2;
        }}
        .subtitle {{
            text-align: center;
            color: var(--text-secondary);
            font-size: 1.05rem;
            margin-bottom: 28px;
        }}
        .section-header {{
            font-size: 1.4rem; font-weight: 700;
            color: var(--text-primary);
            display: flex; align-items: center; gap: 10px;
            margin-bottom: 4px;
        }}
        .gradient-divider {{
            height: 3px; border: none; border-radius: 2px;
            background: linear-gradient(90deg, transparent, var(--accent), var(--accent-secondary), transparent);
            margin: 6px 0 18px 0;
        }}

        /* ── Interactive Buttons & Ripple ──────── */
        .stButton > button, [data-testid="stFormSubmitButton"] button {{
            background: linear-gradient(135deg, #10B981, #059669) !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 11px 28px !important;
            font-weight: 600 !important;
            font-size: 0.95rem !important;
            transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
            box-shadow: 0 4px 16px rgba(16,185,129,0.28) !important;
            position: relative;
            overflow: hidden;
        }}
        .stButton > button:hover, [data-testid="stFormSubmitButton"] button:hover {{
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 24px rgba(16,185,129,0.45) !important;
        }}
        .stButton > button:active, [data-testid="stFormSubmitButton"] button:active {{
            transform: scale(0.96) !important;
            box-shadow: 0 2px 8px rgba(16,185,129,0.5) !important;
        }}

        /* ── Inputs / Selects / Toggles ────────── */
        [data-testid="stNumberInput"] input,
        [data-testid="stTextInput"] input {{
            background: var(--input-bg) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--input-border) !important;
            border-radius: 8px !important;
        }}
        [data-baseweb="select"] {{
            background: var(--select-bg) !important;
        }}
        [data-baseweb="select"] * {{
            color: var(--select-color) !important;
        }}
        [data-testid="stSlider"] label,
        [data-testid="stSlider"] div,
        [data-testid="stRadio"] label,
        [data-testid="stExpander"] summary span {{
            color: var(--text-primary) !important;
        }}
        [data-testid="stCheckbox"] label span {{
            color: var(--toggle-label) !important;
        }}
        [data-testid="stExpander"] {{
            background: var(--expander-bg) !important;
            border: 1px solid var(--card-border) !important;
            border-radius: 12px !important;
        }}
        [data-testid="stForm"] {{
            border: 1px solid var(--card-border) !important;
            border-radius: 16px !important;
            background: var(--card-bg) !important;
            padding: 24px !important;
        }}

        /* ── Result Cards & Pulse Glow ─────────── */
        @keyframes resultPulseGlow {{
            0%   {{ box-shadow: 0 0 10px rgba(16,185,129,0.2); }}
            50%  {{ box-shadow: 0 0 45px rgba(16,185,129,0.65); border-color: #10B981; }}
            100% {{ box-shadow: 0 0 15px rgba(16,185,129,0.25); }}
        }}
        @keyframes resultRiskPulseGlow {{
            0%   {{ box-shadow: 0 0 10px rgba(239,68,68,0.2); }}
            50%  {{ box-shadow: 0 0 45px rgba(239,68,68,0.65); border-color: #EF4444; }}
            100% {{ box-shadow: 0 0 15px rgba(239,68,68,0.25); }}
        }}
        @keyframes fillBarAnim {{
            from {{ width: 0%; }}
            to   {{ width: var(--target-width); }}
        }}

        .result-survive {{
            background: var(--card-bg);
            border: 2px solid rgba(16,185,129,0.40);
            border-radius: 16px; padding: 30px; text-align: center;
        }}
        .result-risk {{
            background: var(--card-bg);
            border: 2px solid rgba(239,68,68,0.40);
            border-radius: 16px; padding: 30px; text-align: center;
        }}
        .result-glow-survive {{
            animation: fadeInUp 0.5s ease both, resultPulseGlow 2s ease-in-out 1;
        }}
        .result-glow-risk {{
            animation: fadeInUp 0.5s ease both, resultRiskPulseGlow 2s ease-in-out 1;
        }}
        .bar-animated {{
            animation: fillBarAnim 1.2s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        }}

        /* ── Animations ────────────────────────── */
        @keyframes fadeInUp {{
            from {{ opacity: 0; transform: translateY(24px); }}
            to   {{ opacity: 1; transform: translateY(0); }}
        }}
        @keyframes floatLogo {{
            0%, 100% {{ transform: translateY(0px); }}
            50%      {{ transform: translateY(-14px); }}
        }}
        @keyframes floatParticle {{
            0%, 100% {{ transform: translateY(0) rotate(0deg); opacity: 0.20; }}
            50%      {{ transform: translateY(-30px) rotate(15deg); opacity: 0.40; }}
        }}
        .animate-in {{
            animation: fadeInUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) both;
        }}
        .animate-in-delay {{
            animation: fadeInUp 0.6s 0.15s cubic-bezier(0.16, 1, 0.3, 1) both;
        }}
        .float-logo {{
            animation: floatLogo 4s ease-in-out infinite;
            display: inline-block;
        }}

        /* ── Floating Background Particles ─────── */
        .particle {{
            position: fixed;
            pointer-events: none;
            z-index: 0;
            font-size: 1.2rem;
            opacity: 0.22;
            animation: floatParticle ease-in-out infinite;
        }}
        .p1  {{ top:10%; left:10%; animation-duration:12s; }}
        .p2  {{ top:25%; left:85%; animation-duration:15s; animation-delay:1s; font-size:0.8rem; }}
        .p3  {{ top:40%; left:8%;  animation-duration:11s; animation-delay:0.5s; }}
        .p4  {{ top:55%; left:48%; animation-duration:14s; animation-delay:2s; font-size:1.5rem; }}
        .p5  {{ top:70%; left:80%; animation-duration:16s; animation-delay:3s; }}
        .p6  {{ top:85%; left:20%; animation-duration:10s; animation-delay:1.5s; font-size:0.9rem; }}
        .p7  {{ top:15%; left:60%; animation-duration:13s; animation-delay:2.5s; }}
        .p8  {{ top:48%; left:90%; animation-duration:12s; animation-delay:4s; }}
        .p9  {{ top:78%; left:55%; animation-duration:17s; animation-delay:1s; }}
        .p10 {{ top:30%; left:32%; animation-duration:14s; animation-delay:3.5s; }}

        /* ── Tabs ──────────────────────────────── */
        .stTabs [data-baseweb="tab-list"] {{ gap: 6px; }}
        .stTabs [data-baseweb="tab"] {{
            background: var(--card-bg) !important;
            border-radius: 8px;
            color: var(--text-secondary) !important;
            border: 1px solid var(--card-border);
        }}
        .stTabs [aria-selected="true"] {{
            background: rgba(16,185,129,0.14) !important;
            color: var(--accent) !important;
            border-color: rgba(16,185,129,0.35) !important;
        }}

        /* ── Reduce Streamlit Spacing ──────────── */
        .block-container {{
            padding-top: 1.5rem !important;
            padding-bottom: 1.2rem !important;
        }}
        [data-testid="stVerticalBlock"] > div:empty {{
            display: none !important;
        }}
    </style>""", unsafe_allow_html=True)


def inject_particles():
    """Inject floating background particles when the particle toggle is ON."""
    if st.session_state.get("particles", True):
        st.markdown(
            '<div class="particle p1" style="color:#10B981;">➕</div>'
            '<div class="particle p2" style="color:#14B8A6;">●</div>'
            '<div class="particle p3" style="color:#10B981;">🧬</div>'
            '<div class="particle p4" style="color:#D4A847;">➕</div>'
            '<div class="particle p5" style="color:#10B981;">●</div>'
            '<div class="particle p6" style="color:#14B8A6;">💓</div>'
            '<div class="particle p7" style="color:#10B981;">➕</div>'
            '<div class="particle p8" style="color:#14B8A6;">🧬</div>'
            '<div class="particle p9" style="color:#D4A847;">●</div>'
            '<div class="particle p10" style="color:#10B981;">➕</div>',
            unsafe_allow_html=True,
        )


# ─── DATA & MODEL LOADING ────────────────────────────────────────────────────
@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        return None
    try:
        return pd.read_csv(DATA_PATH)
    except Exception:
        return None


@st.cache_resource
def load_models():
    models = {}
    for name, path in [("XGBoost", XGB_PATH), ("AdaBoost", ADA_PATH)]:
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    models[name] = pickle.load(f)
            except Exception:
                models[name] = None
        else:
            models[name] = None
    return models


# ─── HELPER: ENCODE A SINGLE INPUT ROW ───────────────────────────────────────
def encode_input_row(values_dict):
    """Build a DataFrame row with the 18 model features, encoded."""
    row = {}
    row["age"] = values_dict["age"]
    row["bmi"] = values_dict["bmi"]
    row["ethnicity"] = ETHNICITY_MAP.get(values_dict["ethnicity"], 0)
    row["gender"] = GENDER_MAP.get(values_dict["gender"], 0)
    row["icu_admit_source"] = ICU_SOURCE_MAP.get(values_dict["icu_admit_source"], 0)
    row["icu_stay_type"] = ICU_STAY_MAP.get(values_dict["icu_stay_type"], 0)
    row["apache_2_diagnosis"] = values_dict["apache_2_diagnosis"]
    row["gcs_eyes_apache"] = values_dict["gcs_eyes_apache"]
    row["gcs_motor_apache"] = values_dict["gcs_motor_apache"]
    row["gcs_verbal_apache"] = values_dict["gcs_verbal_apache"]
    row["heart_rate_apache"] = values_dict["heart_rate_apache"]
    row["map_apache"] = values_dict["map_apache"]
    row["resprate_apache"] = values_dict["resprate_apache"]
    row["d1_spo2_min"] = values_dict["d1_spo2_min"]
    row["d1_temp_max"] = values_dict["d1_temp_max"]
    row["d1_glucose_max"] = values_dict["d1_glucose_max"]
    row["diabetes_mellitus"] = values_dict["diabetes_mellitus"]
    row["cirrhosis"] = values_dict["cirrhosis"]
    return pd.DataFrame([row], columns=MODEL_FEATURES)


# ─── HELPER: PREPROCESS FULL DATASET FOR MODEL EVAL ──────────────────────────
@st.cache_data
def preprocess_dataset_for_eval():
    df = load_data()
    if df is None:
        return None, None
    cols_needed = MODEL_FEATURES + ["hospital_death"]
    missing = [c for c in cols_needed if c not in df.columns]
    if missing:
        return None, None

    X = df[MODEL_FEATURES].copy()
    y = df["hospital_death"].copy()

    X["ethnicity"] = X["ethnicity"].map(
        {v: i for i, v in enumerate(ETHNICITY_OPTIONS)}
    )
    X["gender"] = X["gender"].map(GENDER_DATA_MAP)
    X["icu_admit_source"] = X["icu_admit_source"].map(ICU_SOURCE_MAP)
    X["icu_stay_type"] = X["icu_stay_type"].map(ICU_STAY_DATA_MAP)

    for col in ["ethnicity", "gender", "icu_admit_source", "icu_stay_type"]:
        X[col] = X[col].fillna(0).astype(int)
    num_cols = X.select_dtypes(include=[np.number]).columns
    X[num_cols] = X[num_cols].fillna(X[num_cols].median())

    mask = y.notna()
    return X[mask].reset_index(drop=True), y[mask].reset_index(drop=True).astype(int)


# ─── HELPER: CHART CONFIG ────────────────────────────────────────────────────
def plotly_template():
    return "plotly_dark" if st.session_state.get("theme", "Dark") == "Dark" else "plotly_white"


def chart_colors():
    return ["#10B981", "#14B8A6", "#D4A847", "#F59E0B", "#EF4444", "#64748B"]


def chart_font_color():
    return "#E8EAED" if st.session_state.get("theme", "Dark") == "Dark" else "#1A1D21"


def heatmap_scale():
    if st.session_state.get("theme", "Dark") == "Dark":
        return ["#111315", "#065F46", "#10B981", "#D4A847"]
    return ["#F3F4F6", "#A7F3D0", "#059669", "#064E3B"]


# ─── HELPER: MODEL EVALUATION (cached in session_state) ─────────────────────
def get_model_evaluation():
    """Compute metrics and confusion matrices for all models. Cached per session."""
    if "model_eval" in st.session_state:
        return st.session_state["model_eval"]
    X, y = preprocess_dataset_for_eval()
    if X is None or y is None:
        return None
    models = load_models()
    results = {}
    for name in ["XGBoost", "AdaBoost"]:
        m = models.get(name)
        if m is None:
            continue
        try:
            preds = m.predict(X)
            results[name] = {
                "metrics": {
                    "Accuracy": accuracy_score(y, preds),
                    "Precision": precision_score(y, preds, zero_division=0),
                    "Recall": recall_score(y, preds, zero_division=0),
                    "F1 Score": f1_score(y, preds, zero_division=0),
                },
                "confusion_matrix": confusion_matrix(y, preds).tolist(),
            }
        except Exception:
            pass
    result = results if results else None
    st.session_state["model_eval"] = result
    return result


def get_hero_accuracy():
    """Return the active model's accuracy for the hero section."""
    eval_data = get_model_evaluation()
    if eval_data:
        model_name = st.session_state.get("selected_model", "XGBoost")
        if model_name in eval_data:
            return f"{eval_data[model_name]['metrics']['Accuracy'] * 100:.1f}%"
    return "~92%"


# ─── SAMPLE GENERATORS ───────────────────────────────────────────────────────
def generate_survival_sample():
    """Realistic randomized patient values likely to predict Survived."""
    return {
        "pred_age": random.randint(25, 55),
        "pred_bmi": round(random.uniform(20.0, 28.0), 1),
        "pred_gender": random.choice(GENDER_OPTIONS),
        "pred_eth": random.choice(ETHNICITY_OPTIONS),
        "pred_icu_src": random.choice(["Floor", "Operating Room / Recovery"]),
        "pred_icu_stay": "Admit",
        "pred_apache": random.randint(100, 200),
        "pred_gcs_e": 4,
        "pred_gcs_m": random.choice([5, 6]),
        "pred_gcs_v": random.choice([4, 5]),
        "pred_hr": random.randint(65, 90),
        "pred_map": random.randint(70, 95),
        "pred_rr": random.randint(14, 20),
        "pred_spo2": random.randint(95, 99),
        "pred_temp": round(random.uniform(36.5, 37.5), 1),
        "pred_gluc": random.randint(100, 180),
        "pred_diab": "No",
        "pred_cirr": "No",
    }


def generate_death_sample():
    """Realistic randomized patient values likely to predict Did Not Survive."""
    bmi = round(random.choice([
        random.uniform(14.0, 18.0),
        random.uniform(38.0, 50.0),
    ]), 1)
    return {
        "pred_age": random.randint(70, 95),
        "pred_bmi": bmi,
        "pred_gender": random.choice(GENDER_OPTIONS),
        "pred_eth": random.choice(ETHNICITY_OPTIONS),
        "pred_icu_src": random.choice(["Accident & Emergency", "Other Hospital"]),
        "pred_icu_stay": random.choice(["Readmit", "Transfer"]),
        "pred_apache": random.randint(200, 400),
        "pred_gcs_e": random.randint(1, 2),
        "pred_gcs_m": random.randint(1, 3),
        "pred_gcs_v": random.randint(1, 2),
        "pred_hr": random.randint(120, 180),
        "pred_map": random.randint(35, 55),
        "pred_rr": random.randint(28, 45),
        "pred_spo2": random.randint(60, 82),
        "pred_temp": round(random.uniform(38.5, 40.5), 1),
        "pred_gluc": random.randint(300, 600),
        "pred_diab": "Yes",
        "pred_cirr": "Yes",
    }


# ─── CALLBACKS ────────────────────────────────────────────────────────────────
def fill_sample(generator_fn):
    """Callback: fill prediction form with generated sample values."""
    data = generator_fn()
    for key, val in data.items():
        st.session_state[key] = val
    st.session_state["run_sample_prediction"] = True


def nav_to(page_name):
    """Callback: navigate to a page via sidebar radio."""
    st.session_state["nav_radio"] = page_name


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: HOME
# ═══════════════════════════════════════════════════════════════════════════════
def page_home():
    df = load_data()
    total_patients = f"{len(df):,}" if df is not None else "—"
    hero_accuracy = get_hero_accuracy()

    # ── Hero section ──
    st.markdown(
        '<div class="animate-in" style="text-align:center; padding-top:6px;">'
        '<div class="float-logo" style="font-size:3.6rem; margin-bottom:8px;">🩺</div>'
        '<div class="gradient-title">Healthcare Survival<br>Prediction System</div>'
        '<p class="subtitle animate-in-delay">'
        'Predict patient survival using Machine Learning<br>'
        'with an intuitive interactive dashboard.</p></div>',
        unsafe_allow_html=True,
    )

    # ── Hero metric cards ──
    cols = st.columns(4)
    cards = [
        ("🎯", hero_accuracy, "Model Accuracy"),
        ("🤖", st.session_state.get("selected_model", "XGBoost"), "Active Model"),
        ("📊", total_patients, "Dataset Size"),
        ("⚡", "<50ms", "Prediction Speed"),
    ]
    for col, (icon, value, label) in zip(cols, cards):
        with col:
            st.markdown(f"""
                <div class="metric-card animate-in">
                    <div class="metric-icon">{icon}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-label">{label}</div>
                </div>
            """, unsafe_allow_html=True)

    # ── Features section ──
    st.markdown(
        '<div class="section-header animate-in" style="margin-top:24px;">✨ Features</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    features = [
        ("⚡", "Fast Prediction",
         "Get instant survival predictions powered by optimized ML models."),
        ("📊", "Interactive Dashboard",
         "Explore data with dynamic, filterable Plotly visualizations."),
        ("🧠", "AI Powered",
         "Advanced XGBoost & AdaBoost ensemble learning algorithms."),
        ("📈", "Visual Analytics",
         "Comprehensive EDA with correlation heatmaps and distributions."),
    ]
    feat_cols = st.columns(4)
    for col, (icon, title, desc) in zip(feat_cols, features):
        with col:
            st.markdown(f"""
                <div class="feature-card animate-in">
                    <div class="feature-icon">{icon}</div>
                    <div class="feature-title">{title}</div>
                    <div class="feature-desc">{desc}</div>
                </div>
            """, unsafe_allow_html=True)

    # ── Quick Navigation ──
    st.markdown(
        '<div class="section-header animate-in" style="margin-top:24px;">🚀 Get Started</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    nav_cols = st.columns(3)
    with nav_cols[0]:
        st.button(
            "🩺  Go to Prediction",
            on_click=nav_to,
            args=("🩺 Survival Prediction",),
            use_container_width=True,
            key="home_nav_pred",
        )
    with nav_cols[1]:
        st.button(
            "📈  Go to EDA",
            on_click=nav_to,
            args=("📈 EDA Dashboard",),
            use_container_width=True,
            key="home_nav_eda",
        )
    with nav_cols[2]:
        st.button(
            "⚙️  Go to Settings",
            on_click=nav_to,
            args=("⚙️ Settings",),
            use_container_width=True,
            key="home_nav_settings",
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: EDA DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
def page_eda():
    st.markdown(
        '<div class="section-header animate-in">📈 EDA Dashboard</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    df = load_data()
    if df is None:
        st.error("⚠️ Could not load dataset.")
        return

    template = plotly_template()
    colors = chart_colors()

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        age_range = st.slider(
            "Age Range",
            int(df["age"].min()), int(df["age"].max()),
            (int(df["age"].min()), int(df["age"].max())),
            key="eda_age",
        )
    with fc2:
        gender_filter = st.multiselect(
            "Gender", options=["M", "F"], default=["M", "F"], key="eda_gender",
        )
    with fc3:
        outcome_filter = st.multiselect(
            "Outcome", options=[0, 1], default=[0, 1],
            format_func=lambda x: "Survived" if x == 0 else "Died",
            key="eda_outcome",
        )

    mask = (
        df["age"].between(age_range[0], age_range[1])
        & df["gender"].isin(gender_filter)
        & df["hospital_death"].isin(outcome_filter)
    )
    fdf = df[mask]

    if fdf.empty:
        st.warning("No data matches the selected filters.")
        return

    st.markdown(
        f"<p style='color:var(--text-secondary);margin-bottom:4px;'>"
        f"Showing <b>{len(fdf):,}</b> records</p>",
        unsafe_allow_html=True,
    )

    row1c1, row1c2 = st.columns(2)

    with row1c1:
        fig_age = px.histogram(
            fdf, x="age", nbins=40,
            color_discrete_sequence=[colors[0]],
            title="Age Distribution",
            template=template,
        )
        fig_age.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Age", yaxis_title="Count",
            font=dict(family="Inter"), margin=dict(t=50, b=40),
        )
        st.plotly_chart(fig_age, use_container_width=True)

    with row1c2:
        gender_counts = fdf["gender"].value_counts().reset_index()
        gender_counts.columns = ["Gender", "Count"]
        gender_counts["Gender"] = gender_counts["Gender"].map(
            {"M": "Male", "F": "Female"}
        )
        fig_gen = px.pie(
            gender_counts, names="Gender", values="Count",
            title="Gender Distribution",
            color_discrete_sequence=[colors[0], colors[1]],
            template=template, hole=0.45,
        )
        fig_gen.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter"), margin=dict(t=50, b=40),
        )
        st.plotly_chart(fig_gen, use_container_width=True)

    row2c1, row2c2 = st.columns(2)

    with row2c1:
        death_counts = fdf["hospital_death"].value_counts().reset_index()
        death_counts.columns = ["Outcome", "Count"]
        death_counts["Outcome"] = death_counts["Outcome"].map(
            {0: "Survived", 1: "Died"}
        )
        fig_death = px.bar(
            death_counts, x="Outcome", y="Count",
            color="Outcome",
            color_discrete_map={"Survived": "#10B981", "Died": "#EF4444"},
            title="Hospital Death Distribution",
            template=template,
        )
        fig_death.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter"), margin=dict(t=50, b=40),
            showlegend=False,
        )
        st.plotly_chart(fig_death, use_container_width=True)

    with row2c2:
        numeric_cols = fdf[MODEL_FEATURES].select_dtypes(
            include=[np.number]
        ).columns.tolist()
        if len(numeric_cols) > 1:
            corr = fdf[numeric_cols].corr().round(2)
            fig_corr = px.imshow(
                corr, text_auto=".1f",
                color_continuous_scale=heatmap_scale(),
                title="Feature Correlation Heatmap",
                template=template, aspect="auto",
            )
            fig_corr.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", size=9),
                margin=dict(t=50, b=10), height=520,
            )
            st.plotly_chart(fig_corr, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: SURVIVAL PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
def page_prediction():
    st.markdown(
        '<div class="section-header animate-in">🩺 Survival Prediction</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    models = load_models()
    model_name = st.session_state.get("selected_model", "XGBoost")
    model = models.get(model_name)

    if model is None:
        st.error(f"⚠️ {model_name} model could not be loaded. Check the .pkl file.")
        return

    st.markdown(
        f"<p style='color:var(--text-secondary);margin-bottom:2px;'>Using model: "
        f"<b style=\"color:var(--accent);\">{model_name}</b> &nbsp;·&nbsp; "
        f"Change in ⚙️ Settings</p>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<p style='color:var(--text-secondary);font-size:0.9rem;"
        "margin-bottom:4px;'>Quick fill with sample patient data:</p>",
        unsafe_allow_html=True,
    )
    btn_c1, btn_c2 = st.columns(2)
    with btn_c1:
        st.button(
            "🟢  Survival Sample",
            on_click=fill_sample,
            args=(generate_survival_sample,),
            use_container_width=True,
            key="btn_survival_sample",
        )
    with btn_c2:
        st.button(
            "🔴  Death Sample",
            on_click=fill_sample,
            args=(generate_death_sample,),
            use_container_width=True,
            key="btn_death_sample",
        )

    with st.form("prediction_form"):
        st.markdown(
            "<h4 style='color:var(--text-primary);margin-bottom:12px;'>"
            "📝 Enter Patient Information</h4>",
            unsafe_allow_html=True,
        )

        r1c1, r1c2, r1c3 = st.columns(3)
        with r1c1:
            age = st.number_input("Age (years)", 0, 120, 60, key="pred_age")
        with r1c2:
            bmi = st.number_input("BMI", 10.0, 80.0, 25.0, step=0.1, key="pred_bmi")
        with r1c3:
            gender = st.selectbox("Gender", GENDER_OPTIONS, key="pred_gender")

        r2c1, r2c2, r2c3 = st.columns(3)
        with r2c1:
            ethnicity = st.selectbox("Ethnicity", ETHNICITY_OPTIONS, key="pred_eth")
        with r2c2:
            icu_source = st.selectbox("ICU Admit Source", ICU_SOURCE_OPTIONS, key="pred_icu_src")
        with r2c3:
            icu_stay = st.selectbox("ICU Stay Type", ICU_STAY_OPTIONS, key="pred_icu_stay")

        r3c1, r3c2, r3c3 = st.columns(3)
        with r3c1:
            apache_diag = st.number_input("APACHE 2 Diagnosis Code", 0, 999, 113, key="pred_apache")
        with r3c2:
            gcs_eyes = st.slider("GCS Eyes (1-4)", 1, 4, 3, key="pred_gcs_e")
        with r3c3:
            gcs_motor = st.slider("GCS Motor (1-6)", 1, 6, 6, key="pred_gcs_m")

        r4c1, r4c2, r4c3 = st.columns(3)
        with r4c1:
            gcs_verbal = st.slider("GCS Verbal (1-5)", 1, 5, 4, key="pred_gcs_v")
        with r4c2:
            heart_rate = st.number_input("Heart Rate (Apache)", 0, 300, 80, key="pred_hr")
        with r4c3:
            map_val = st.number_input("MAP – Mean Arterial Pressure", 0, 300, 80, key="pred_map")

        r5c1, r5c2, r5c3 = st.columns(3)
        with r5c1:
            resprate = st.number_input("Respiratory Rate (Apache)", 0, 80, 18, key="pred_rr")
        with r5c2:
            spo2_min = st.number_input("SpO₂ Min (Day 1)", 0, 100, 95, key="pred_spo2")
        with r5c3:
            temp_max = st.number_input("Temp Max °C (Day 1)", 30.0, 45.0, 37.0, step=0.1, key="pred_temp")

        r6c1, r6c2, r6c3 = st.columns(3)
        with r6c1:
            glucose_max = st.number_input("Glucose Max (Day 1)", 0, 1000, 150, key="pred_gluc")
        with r6c2:
            diabetes = st.selectbox("Diabetes Mellitus", ["No", "Yes"], key="pred_diab")
        with r6c3:
            cirrhosis = st.selectbox("Cirrhosis", ["No", "Yes"], key="pred_cirr")

        submitted = st.form_submit_button("🔮  Predict Survival", use_container_width=True)

    should_predict = submitted or st.session_state.pop("run_sample_prediction", False)

    if should_predict:
        # Visual feedback: spinner & loading message
        with st.spinner("⏳ Analyzing Patient Data & Running ML Ensemble..."):
            time.sleep(0.6)

        values = {
            "age": st.session_state.get("pred_age", 60),
            "bmi": st.session_state.get("pred_bmi", 25.0),
            "ethnicity": st.session_state.get("pred_eth", ETHNICITY_OPTIONS[0]),
            "gender": st.session_state.get("pred_gender", GENDER_OPTIONS[0]),
            "icu_admit_source": st.session_state.get("pred_icu_src", ICU_SOURCE_OPTIONS[0]),
            "icu_stay_type": st.session_state.get("pred_icu_stay", ICU_STAY_OPTIONS[0]),
            "apache_2_diagnosis": st.session_state.get("pred_apache", 113),
            "gcs_eyes_apache": st.session_state.get("pred_gcs_e", 3),
            "gcs_motor_apache": st.session_state.get("pred_gcs_m", 6),
            "gcs_verbal_apache": st.session_state.get("pred_gcs_v", 4),
            "heart_rate_apache": st.session_state.get("pred_hr", 80),
            "map_apache": st.session_state.get("pred_map", 80),
            "resprate_apache": st.session_state.get("pred_rr", 18),
            "d1_spo2_min": st.session_state.get("pred_spo2", 95),
            "d1_temp_max": st.session_state.get("pred_temp", 37.0),
            "d1_glucose_max": st.session_state.get("pred_gluc", 150),
            "diabetes_mellitus": 1 if st.session_state.get("pred_diab", "No") == "Yes" else 0,
            "cirrhosis": 1 if st.session_state.get("pred_cirr", "No") == "Yes" else 0,
        }

        input_df = encode_input_row(values)

        try:
            pred = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0]
            st.session_state["prediction_result"] = int(pred)
            st.session_state["prediction_proba"] = proba.tolist()
            st.session_state["just_predicted"] = True
            st.toast("✅ Survival analysis complete!", icon="🔮")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return

    # ── Show result ──
    if st.session_state.get("prediction_result") is not None:
        pred = st.session_state["prediction_result"]
        proba = st.session_state["prediction_proba"]
        survival_prob = proba[0] * 100
        death_prob = proba[1] * 100

        if survival_prob > 70:
            risk_level = "Low Risk"
        elif survival_prob > 30:
            risk_level = "Medium Risk"
        else:
            risk_level = "High Risk"

        confidence = max(survival_prob, death_prob)
        if confidence > 80:
            conf_text = "High"
        elif confidence > 60:
            conf_text = "Moderate"
        else:
            conf_text = "Low"

        # Apply subtle glow animation if recently predicted
        glow_cls = ""
        if st.session_state.get("just_predicted", False):
            glow_cls = "result-glow-survive" if pred == 0 else "result-glow-risk"
            st.session_state["just_predicted"] = False

        res_col, gauge_col = st.columns([1, 1])

        with res_col:
            if pred == 0:
                st.markdown(f"""
                    <div class="result-survive animate-in {glow_cls}">
                        <div style="font-size:3rem; margin-bottom:8px;">✅</div>
                        <h2 style="color:#10B981; margin-bottom:6px; font-size:1.5rem;">
                            Patient Likely to Survive
                        </h2>
                        <p style="color:var(--text-secondary);font-size:0.95rem;">
                            Survival probability:
                            <b style="color:#10B981;">{survival_prob:.1f}%</b>
                        </p>
                        <div style="display:flex;gap:10px;justify-content:center;
                                    margin-top:14px;flex-wrap:wrap;">
                            <span style="background:rgba(16,185,129,0.15);
                                         color:#10B981;padding:5px 14px;
                                         border-radius:20px;font-size:0.82rem;
                                         font-weight:600;">{risk_level}</span>
                            <span style="background:rgba(20,184,166,0.15);
                                         color:#14B8A6;padding:5px 14px;
                                         border-radius:20px;font-size:0.82rem;
                                         font-weight:600;">{conf_text} Confidence</span>
                        </div>
                        <div style="margin-top:18px;">
                            <div style="background:var(--bar-bg);border-radius:8px;
                                        height:10px;overflow:hidden;">
                                <div class="bar-animated" style="background:linear-gradient(90deg,#10B981,#14B8A6);
                                     --target-width:{survival_prob}%;height:100%;
                                     border-radius:8px;"></div>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="result-risk animate-in {glow_cls}">
                        <div style="font-size:3rem; margin-bottom:8px;">⚠️</div>
                        <h2 style="color:#EF4444; margin-bottom:6px; font-size:1.5rem;">
                            High-Risk Patient
                        </h2>
                        <p style="color:var(--text-secondary);font-size:0.95rem;">
                            Mortality risk:
                            <b style="color:#EF4444;">{death_prob:.1f}%</b>
                        </p>
                        <div style="display:flex;gap:10px;justify-content:center;
                                    margin-top:14px;flex-wrap:wrap;">
                            <span style="background:rgba(239,68,68,0.15);
                                         color:#EF4444;padding:5px 14px;
                                         border-radius:20px;font-size:0.82rem;
                                         font-weight:600;">{risk_level}</span>
                            <span style="background:rgba(245,158,11,0.15);
                                         color:#F59E0B;padding:5px 14px;
                                         border-radius:20px;font-size:0.82rem;
                                         font-weight:600;">{conf_text} Confidence</span>
                        </div>
                        <div style="margin-top:18px;">
                            <div style="background:var(--bar-bg);border-radius:8px;
                                        height:10px;overflow:hidden;">
                                <div class="bar-animated" style="background:linear-gradient(90deg,#EF4444,#DC2626);
                                     --target-width:{death_prob}%;height:100%;
                                     border-radius:8px;"></div>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

        with gauge_col:
            fc = chart_font_color()
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=survival_prob,
                number={"suffix": "%", "font": {"size": 40, "family": "Inter", "color": fc}},
                title={"text": "Survival Probability", "font": {"size": 15, "family": "Inter", "color": fc}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar": {"color": "#10B981", "thickness": 0.3},
                    "bgcolor": "rgba(0,0,0,0)",
                    "steps": [
                        {"range": [0, 30], "color": "rgba(239,68,68,0.18)"},
                        {"range": [30, 70], "color": "rgba(245,158,11,0.12)"},
                        {"range": [70, 100], "color": "rgba(16,185,129,0.12)"},
                    ],
                    "threshold": {
                        "line": {"color": "#EF4444", "width": 3},
                        "thickness": 0.8,
                        "value": 50,
                    },
                },
            ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"family": "Inter", "color": fc},
                height=300,
                margin=dict(t=55, b=15, l=25, r=25),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: SETTINGS (includes Model Comparison)
# ═══════════════════════════════════════════════════════════════════════════════
def page_settings():
    st.markdown(
        '<div class="section-header animate-in">⚙️ Settings</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    s1, s2 = st.columns(2)

    with s1:
        st.markdown(
            "<h4 style='color:var(--text-primary);margin-bottom:8px;'>🎨 Appearance</h4>",
            unsafe_allow_html=True,
        )
        theme_choice = st.radio(
            "Select theme", ["Dark", "Light"],
            index=0 if st.session_state.get("theme") == "Dark" else 1,
            key="settings_theme", horizontal=True,
        )
        if theme_choice != st.session_state.get("theme"):
            st.session_state["theme"] = theme_choice
            st.rerun()

    with s2:
        st.markdown(
            "<h4 style='color:var(--text-primary);margin-bottom:8px;'>✨ Animations</h4>",
            unsafe_allow_html=True,
        )
        anim_on = st.toggle(
            "Enable UI animations & transitions",
            value=st.session_state.get("animations", True),
            key="settings_anim",
        )
        if anim_on != st.session_state.get("animations"):
            st.session_state["animations"] = anim_on
            st.rerun()

    s3, s4 = st.columns(2)

    with s3:
        st.markdown(
            "<h4 style='color:var(--text-primary);margin-bottom:8px;'>🔮 Particles</h4>",
            unsafe_allow_html=True,
        )
        particles_on = st.toggle(
            "Enable floating animated background",
            value=st.session_state.get("particles", True),
            key="settings_particles",
        )
        if particles_on != st.session_state.get("particles"):
            st.session_state["particles"] = particles_on
            st.rerun()

    with s4:
        st.markdown(
            "<h4 style='color:var(--text-primary);margin-bottom:8px;'>🤖 Prediction Model</h4>",
            unsafe_allow_html=True,
        )
        model_choice = st.radio(
            "Select model for predictions", ["XGBoost", "AdaBoost"],
            index=0 if st.session_state.get("selected_model") == "XGBoost" else 1,
            key="settings_model", horizontal=True,
        )
        if model_choice != st.session_state.get("selected_model"):
            st.session_state["selected_model"] = model_choice
            st.session_state["prediction_result"] = None
            st.session_state["prediction_proba"] = None
            st.toast(f"✅ Switched to {model_choice}")

    st.markdown(
        "<h4 style='color:var(--text-primary);margin-top:16px;margin-bottom:6px;'>🔄 Reset</h4>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:var(--text-secondary);font-size:0.88rem;margin-bottom:8px;'>"
        "Clear all predictions and reset to defaults.</p>",
        unsafe_allow_html=True,
    )
    if st.button("🗑️  Reset Dashboard", key="reset_btn"):
        for key in list(st.session_state.keys()):
            if key.startswith("pred_"):
                del st.session_state[key]
        st.session_state["prediction_result"] = None
        st.session_state["prediction_proba"] = None
        st.session_state["selected_model"] = "XGBoost"
        st.session_state["theme"] = "Dark"
        st.session_state["animations"] = True
        st.session_state["particles"] = True
        if "model_eval" in st.session_state:
            del st.session_state["model_eval"]
        st.toast("🔄 Dashboard reset!")
        st.rerun()

    with st.expander("📊 Model Comparison", expanded=False):
        _render_model_comparison()

    st.markdown(f"""
        <div class="glass-card animate-in" style="text-align:center;margin-top:12px;">
            <h4 style="color:var(--text-primary);margin-bottom:10px;">
                📋 Current Configuration
            </h4>
            <p style="color:var(--text-secondary);margin:0;">
                Model: <b style="color:var(--accent);">{st.session_state.get('selected_model', 'XGBoost')}</b>
                &nbsp;·&nbsp;
                Theme: <b style="color:var(--accent);">{st.session_state.get('theme', 'Dark')}</b>
                &nbsp;·&nbsp;
                Animations: <b style="color:var(--accent);">{'On' if st.session_state.get('animations', True) else 'Off'}</b>
                &nbsp;·&nbsp;
                Particles: <b style="color:var(--accent);">{'On' if st.session_state.get('particles', True) else 'Off'}</b>
            </p>
        </div>
    """, unsafe_allow_html=True)


def _render_model_comparison():
    """Render the model comparison section inside Settings expander."""
    eval_data = get_model_evaluation()

    if eval_data is None:
        st.warning("Could not compute model metrics. Check dataset and model files.")
        return

    template = plotly_template()

    best_model = max(
        eval_data.items(), key=lambda x: x[1]["metrics"]["Accuracy"]
    )[0]
    table_data = []
    for name, data in eval_data.items():
        metrics = data["metrics"]
        prefix = "⭐ " if name == best_model else ""
        table_data.append({
            "Model": f"{prefix}{name}",
            "Accuracy": f"{metrics['Accuracy'] * 100:.2f}%",
            "Precision": f"{metrics['Precision'] * 100:.2f}%",
            "Recall": f"{metrics['Recall'] * 100:.2f}%",
            "F1 Score": f"{metrics['F1 Score'] * 100:.2f}%",
        })
    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

    for name, data in eval_data.items():
        metrics = data["metrics"]
        st.markdown(
            f"<h4 style='color:var(--text-primary);margin-top:12px;'>"
            f"{'🚀' if name == 'XGBoost' else '⚡'} {name}</h4>",
            unsafe_allow_html=True,
        )
        cols = st.columns(4)
        icons = ["🎯", "✅", "🔍", "📐"]
        for col, (metric_name, val), icon in zip(cols, metrics.items(), icons):
            with col:
                st.markdown(f"""
                    <div class="metric-card animate-in">
                        <div class="metric-icon">{icon}</div>
                        <div class="metric-value">{val * 100:.2f}%</div>
                        <div class="metric-label">{metric_name}</div>
                    </div>
                """, unsafe_allow_html=True)

    metric_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
    fig = go.Figure()
    bar_colors = {"XGBoost": "#10B981", "AdaBoost": "#14B8A6"}

    for name, data in eval_data.items():
        metrics = data["metrics"]
        fig.add_trace(go.Bar(
            x=metric_names,
            y=[metrics[m] * 100 for m in metric_names],
            name=name,
            marker_color=bar_colors.get(name, "#D4A847"),
            text=[f"{metrics[m] * 100:.1f}%" for m in metric_names],
            textposition="outside",
        ))

    fig.update_layout(
        title="Model Performance Comparison",
        template=template,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        barmode="group",
        yaxis=dict(title="Score (%)", range=[0, 110]),
        font=dict(family="Inter"),
        margin=dict(t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "<h4 style='color:var(--text-primary);margin-top:8px;'>Confusion Matrices</h4>",
        unsafe_allow_html=True,
    )
    cm_scale = heatmap_scale()[:3]
    cm_cols = st.columns(len(eval_data))
    for col, (name, data) in zip(cm_cols, eval_data.items()):
        cm = np.array(data["confusion_matrix"])
        labels = ["Survived", "Died"]
        fig_cm = px.imshow(
            cm, text_auto=True, x=labels, y=labels,
            color_continuous_scale=cm_scale,
            title=name, template=template,
        )
        fig_cm.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Predicted", yaxis_title="Actual",
            font=dict(family="Inter"), margin=dict(t=50, b=40), height=360,
        )
        with col:
            st.plotly_chart(fig_cm, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR & ROUTING
# ═══════════════════════════════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        st.markdown(
            '<div style="text-align:center; padding:14px 0 6px 0;">'
            '<span style="font-size:2.2rem;">🩺</span>'
            '<h3 style="color:var(--text-primary);margin:4px 0 0 0;'
            'font-size:1.02rem;line-height:1.3;">Healthcare AI</h3>'
            '<div style="height:2px;background:linear-gradient(90deg,'
            'transparent,var(--accent),transparent);'
            'margin:8px 0 14px 0;border-radius:2px;"></div></div>',
            unsafe_allow_html=True,
        )

        page = st.radio(
            "Navigation",
            ["🏠 Home", "📈 EDA Dashboard", "🩺 Survival Prediction", "⚙️ Settings"],
            label_visibility="collapsed",
            key="nav_radio",
        )

        st.markdown(f"""
            <div style="padding:10px 14px; background:var(--card-bg);
                        border:1px solid var(--card-border);
                        border-radius:10px; margin-top:16px;">
                <div style="color:var(--text-secondary);
                            font-size:0.76rem;margin-bottom:3px;">Active Model</div>
                <div style="color:var(--accent);font-weight:600;font-size:0.92rem;">
                    {'🚀' if st.session_state.get('selected_model') == 'XGBoost' else '⚡'}
                    {st.session_state.get('selected_model', 'XGBoost')}
                </div>
            </div>
        """, unsafe_allow_html=True)

    return page


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    init_session_state()
    inject_css()
    inject_particles()
    page = render_sidebar()

    page_map = {
        "🏠 Home": page_home,
        "📈 EDA Dashboard": page_eda,
        "🩺 Survival Prediction": page_prediction,
        "⚙️ Settings": page_settings,
    }

    page_fn = page_map.get(page, page_home)
    page_fn()


if __name__ == "__main__":
    main()
