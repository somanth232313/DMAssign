# ==============================================================================
# FILE 2: app.py (This is your Streamlit app)
#
# Objective: Load the saved 'model.pkl' and use it to make live predictions
# ==============================================================================

import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --- 1. Load the Saved Model ---
@st.cache_resource  # Caches the model so it doesn't reload on every interaction
def load_model():
    """Loads the pickled model pipeline."""
    try:
        with open('model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("Model file 'model.pkl' not found. Please place it in the same directory.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# --- 2. Page Configuration ---
st.set_page_config(
    page_title="🚢 Titanic Survival Prediction",
    page_icon="🚢",
    layout="centered"
)
st.title("🚢 Titanic Survival Prediction")
st.markdown("""
This app uses the XGBoost model you trained in Google Colab to predict
whether a passenger would have survived the Titanic disaster.
""")

# --- 3. User Input Form ---
if model:
    with st.form("prediction_form"):
        st.header("Enter Passenger Details")
        
        # Split layout into columns for a cleaner look
        col1, col2 = st.columns(2)
        
        with col1:
            pclass = st.selectbox("Passenger Class", options=[1, 2, 3], format_func=lambda x: f"{x}st Class")
            sex = st.selectbox("Sex", options=['male', 'female'])
            age = st.slider("Age", min_value=0, max_value=100, value=30, step=1)
            sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10, value=0, step=1)

        with col2:
            parch = st.number_input("Parents/Children Aboard", min_value=0, max_value=10, value=0, step=1)
            fare = st.number_input("Fare Paid ($)", min_value=0.0, max_value=600.0, value=50.0, step=1.0)
            embarked = st.selectbox("Port of Embarkation", 
                                    options=['S', 'C', 'Q'], 
                                    format_func=lambda x: {'S':'Southampton', 'C':'Cherbourg', 'Q':'Queenstown'}[x])
        
        # Submit button
        st.write("") # Add a little space
        submitted = st.form_submit_button("Predict Survival", use_container_width=True)

    # --- 4. Prediction Logic ---
    if submitted:
        # Create a DataFrame from the inputs
        # The column names MUST match those used during training
        input_data = pd.DataFrame({
            'pclass': [pclass],
            'sex': [sex],
            'age': [age],
            'sibsp': [sibsp],
            'parch': [parch],
            'fare': [fare],
            'embarked': [embarked]
            # Note: We don't need to provide all the *original* columns,
            # just the ones our 'preprocessor' is set up to handle.
        })
        
        st.subheader("Prediction Result")
        with st.spinner("Analyzing..."):
            # Make prediction
            try:
                prediction = model.predict(input_data)
                prediction_proba = model.predict_proba(input_data)
                
                # Get the probabilities
                prob_survived = prediction_proba[0][1]
                prob_not_survived = prediction_proba[0][0]

                # Display result
                if prediction[0] == 1:
                    st.success(f"**Prediction: SURVIVED** (Probability: {prob_survived:.1%})")
                    st.balloons()
                else:
                    st.error(f"**Prediction: DID NOT SURVIVE** (Probability: {prob_not_survived:.1%})")
                
                st.write("---")
                st.write(f"Full Probability: `{prob_survived:.1%}` Survived vs. `{prob_not_survived:.1%}` Did Not Survive")

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
else:
    st.warning("Model could not be loaded. Please ensure 'model.pkl' is present.")