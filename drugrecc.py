import pandas as pd
import streamlit as st
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the pre-trained model and encoders
model = joblib.load('drug_model.pkl')
label_encoder_phase = joblib.load('label_encoder_phase.pkl')
label_encoder_trial_results = joblib.load('label_encoder_trial_results.pkl')

# Page config
st.set_page_config(
    page_title="GENAI FDA Compliance Checker",
    page_icon="💊",
    layout="centered",
    initial_sidebar_state="auto"
)

# Inject some custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
        padding: 2rem;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stTextInput>div>div>input {
        background-color: #ffffff;
    }
    .result-success {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        font-weight: bold;
    }
    .result-fail {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown("<h1 style='text-align: center; color: #4B0082;'>💊 GENAI - FDA Drug Pipeline Compliance Checker</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload your drug pipeline data to check if it complies with FDA regulations using our trained AI model.</p>", unsafe_allow_html=True)
st.markdown("---")

# User Inputs
st.subheader("📄 Enter Drug Information")

drug_name = st.text_input("🔹 Drug Name")
phase = st.selectbox("🔹 Development Phase", ['Preclinical', 'Phase 1', 'Phase 2', 'Phase 3'])
safety_data = st.slider("🛡️ Safety Data Score (1-10)", 1, 10, 5)
efficacy_data = st.slider("💪 Efficacy Data Score (1-10)", 1, 10, 5)
trial_results = st.selectbox("🔹 Trial Outcome", ['Success', 'Failure', 'Adverse Effects'])

# Prepare input data
input_data = pd.DataFrame({
    'drug_name': [drug_name],
    'phase': [phase],
    'safety_data': [safety_data],
    'efficacy_data': [efficacy_data],
    'trial_results': [trial_results]
})

# Helper function for safe label encoding
def safe_transform(encoder, value):
    try:
        return encoder.transform([value])[0]
    except ValueError:
        return -1

# Encode categorical variables
input_data['phase'] = input_data['phase'].apply(lambda x: safe_transform(label_encoder_phase, x))
input_data['trial_results'] = input_data['trial_results'].apply(lambda x: safe_transform(label_encoder_trial_results, x))

# Check Compliance Button
if st.button("✅ Check Compliance"):
    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        st.markdown('<div class="result-success">✅ The drug pipeline is <strong>COMPLIANT</strong> with FDA regulations.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-fail">🚫 The drug pipeline is <strong>NON-COMPLIANT</strong> with FDA regulations.</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown("<small>🔬 Powered by GENAI — Accelerating drug approvals through intelligent compliance checks.</small>", unsafe_allow_html=True)
