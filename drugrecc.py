#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.preprocessing import LabelEncoder
import joblib
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

# Load the pre-trained model (assuming it's already saved as 'drug_model.pkl')
model = joblib.load('drug_model.pkl')

# Create and fit the LabelEncoder on the known labels
label_encoder_phase = LabelEncoder()
label_encoder_trial_results = LabelEncoder()

# Fit the encoder on the possible values of 'phase' and 'trial_results'
label_encoder_phase.fit(['Preclinical', 'Phase 1', 'Phase 2', 'Phase 3'])  # Ensure all phases are included
label_encoder_trial_results.fit(['Success', 'Failure', 'Adverse Effects'])  # Ensure all trial results are included

# Save the fitted encoders for future use
joblib.dump(label_encoder_phase, 'label_encoder_phase.pkl')
joblib.dump(label_encoder_trial_results, 'label_encoder_trial_results.pkl')

# Streamlit interface
st.title("GENAI - FDA Drug Pipeline Compliance Checker")

st.write("""
    Upload the new drug pipeline data, and our tool will analyze it for compliance with FDA regulations.
    Please fill in the information for each column to receive the analysis.
""")

# User input for the drug pipeline
drug_name = st.text_input("Drug Name")
phase = st.selectbox("Phase", ['Preclinical', 'Phase 1', 'Phase 2', 'Phase 3'])
safety_data = st.number_input("Safety Data Score (1-10)", min_value=1, max_value=10)
efficacy_data = st.number_input("Efficacy Data Score (1-10)", min_value=1, max_value=10)
trial_results = st.selectbox("Trial Results", ['Success', 'Failure', 'Adverse Effects'])

# Prepare the input data for prediction
input_data = pd.DataFrame({
    'drug_name': [drug_name],
    'phase': [phase],
    'safety_data': [safety_data],
    'efficacy_data': [efficacy_data],
    'trial_results': [trial_results]
})

# Load the pre-trained label encoders (if they're already saved)
label_encoder_phase = joblib.load('label_encoder_phase.pkl')
label_encoder_trial_results = joblib.load('label_encoder_trial_results.pkl')

# Handle unseen labels gracefully using a helper function
def safe_transform(encoder, value):
    try:
        # Try to transform, if value is unseen, return a default encoding
        return encoder.transform([value])[0]
    except ValueError:
        # Handle unseen label (assign default value, e.g., -1 or any number)
        return -1  # Or some other value you deem appropriate

# Transform the input data with safety checks
input_data['phase'] = input_data['phase'].apply(lambda x: safe_transform(label_encoder_phase, x))
input_data['trial_results'] = input_data['trial_results'].apply(lambda x: safe_transform(label_encoder_trial_results, x))

# Predict compliance with FDA regulations
if st.button('Check Compliance'):
    prediction = model.predict(input_data)  # Drop 'drug_name' as it's not used in prediction
    if prediction == 1:
        st.success("The drug pipeline is **compliant** with FDA regulations.")
    else:
        st.error("The drug pipeline is **non-compliant** with FDA regulations.")


# In[ ]:




