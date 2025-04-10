import pandas as pd
import joblib
import streamlit as st
from sklearn.preprocessing import LabelEncoder

# Load the model and encoders
model = joblib.load('drug_model.pkl')
label_encoder_phase = joblib.load('label_encoder_phase.pkl')
label_encoder_trial_results = joblib.load('label_encoder_trial_results.pkl')

# App title with emoji
st.set_page_config(page_title="FDA Drug Compliance Checker", page_icon="💊", layout="centered")
st.title("💊 GENAI - FDA Drug Pipeline Compliance Checker")

# Custom CSS styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f8ff;
    }
    .block-container {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 2rem 2rem 2rem 2rem;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
    }
    .stButton button {
        background-color: #4B8BBE;
        color: white;
        border-radius: 8px;
        font-weight: bold;
        padding: 10px 20px;
    }
    .stButton button:hover {
        background-color: #306998;
        color: #fff;
    }
    </style>
""", unsafe_allow_html=True)

# App intro
st.markdown("""
Welcome to the **FDA Compliance Checker**.  
Fill in the drug pipeline details below to check if your new drug candidate is likely compliant with FDA regulations.  
""")

st.markdown("---")

# Drug name input
drug_name = st.text_input("🔍 Enter Drug Name", placeholder="e.g., Remedix-Alpha")

# Input layout in columns
col1, col2 = st.columns(2)

with col1:
    phase = st.selectbox("🧪 Select Clinical Phase", ['Preclinical', 'Phase 1', 'Phase 2', 'Phase 3'])
    safety_data = st.slider("🛡️ Safety Data Score", min_value=1, max_value=10, value=5)

with col2:
    trial_results = st.selectbox("📊 Trial Results", ['Success', 'Failure', 'Adverse Effects'])
    efficacy_data = st.slider("💊 Efficacy Data Score", min_value=1, max_value=10, value=5)

st.markdown("---")

# Prepare input data
input_data = pd.DataFrame({
    'drug_name': [drug_name],
    'phase': [phase],
    'safety_data': [safety_data],
    'efficacy_data': [efficacy_data],
    'trial_results': [trial_results]
})

# Helper to handle unseen labels
def safe_transform(encoder, value):
    try:
        return encoder.transform([value])[0]
    except ValueError:
        return -1  # For unseen labels

# Apply label encoding
input_data['phase'] = input_data['phase'].apply(lambda x: safe_transform(label_encoder_phase, x))
input_data['trial_results'] = input_data['trial_results'].apply(lambda x: safe_transform(label_encoder_trial_results, x))

# Button to make prediction
if st.button('🚀 Check Compliance'):
    if drug_name.strip() == "":
        st.warning("⚠️ Please enter the drug name.")
    else:
        prediction = model.predict(input_data.drop(columns=['drug_name']))
        st.markdown("---")
        if prediction == 1:
            st.success("✅ The drug pipeline is **compliant** with FDA regulations. Ready to proceed!")
            st.balloons()
        else:
            st.error("❌ The drug pipeline is **non-compliant**. Please review safety or efficacy data.")

# Footer
st.markdown("---")
st.markdown("📢 *This tool uses a pre-trained model and is for research/demonstration purposes only.*")
