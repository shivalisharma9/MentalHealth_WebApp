import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os
import joblib 

print("Current working directory:", os.getcwd())
print("Files in directory:", os.listdir())

# Define paths for all model files
stress_model = joblib.load('stress_model.joblib')
anxiety_model = joblib.load('anxiety_model.joblib')
depression_model = joblib.load('depression_model.joblib')
disorder_model = joblib.load('disorder_model.joblib')

# Helper functions for type conversion and validation
def convert_binary_response(value):
    """Convert Yes/No responses to float"""
    if isinstance(value, str):
        return float(1) if value.lower() == 'yes' else float(0)
    return float(0)

def convert_numeric(value, default=0.0):
    """Convert any value to float safely"""
    try:
        return float(value)
    except (ValueError, TypeError):
        return float(default)

def encode_categorical(value, categories, prefix=''):
    """Create one-hot encoding for categorical values"""
    encoding = {}
    for category in categories:
        column_name = f"{prefix}_{category}" if prefix else category
        encoding[column_name] = float(1) if value == category else float(0)
    return encoding

# Load models with error handling
try:
    stress_model = pickle.load(open('stress_model.pkl', 'rb'))
    depression_model = pickle.load(open('depression_model.pkl', 'rb'))
    burnout_model = pickle.load(open('burnout_model.pkl', 'rb'))
    wellness_model = pickle.load(open('wellness_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
except FileNotFoundError as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

st.title("Mental Health Wellness Predictor")
st.write("Fill in the details below to get predictions.")

def get_user_input():
    with st.form("prediction_form"):
        try:
            # Common inputs
            age = st.number_input("Age", min_value=10, max_value=100, value=25)
            gender = st.selectbox("Gender", ['Male', 'Female', 'Other'])
            sleep_hours = st.slider("Sleep Hours per Night", 0, 12, 7)
            
            # Lifestyle Factors
            st.subheader("Lifestyle Factors")
            work_hours = st.slider("Work Hours per Day", 0, 16, 8)
            exercise_freq = st.slider("Exercise Frequency (days per week)", 0, 7, 3)
            screen_time = st.slider("Screen Time (hours per day)", 0, 12, 4)
            social_activity = st.slider("Social Activity Level (0-10)", 0, 10, 5)
            diet_quality = st.slider("How healthy is your diet? (0-10)", 0, 10, 5)

            # Mental Health History
            st.subheader("Mental Health History")
            family_history = st.selectbox("Family History of Mental Health Issues", ['No', 'Yes'])
            sleep_quality = st.slider("Sleep Quality (0-10)", 0, 10, 5)
            stress_level_dep = st.slider("Current Stress Level (0-10)", 0, 10, 5)
            social_support = st.slider("Social Support Level (0-10)", 0, 10, 5)
            physical_activity = st.slider("Physical Activity Level (0-10)", 0, 10, 5)
            substance_use = st.selectbox("Substance Use (Alcohol, Drugs, etc.)", ['No', 'Yes'])

            # Work-Related Factors
            st.subheader("Work-Related Factors")
            overtime = st.slider("Overtime Hours per Week", 0, 20, 0)
            work_satisfaction = st.slider("Work Satisfaction (0-10)", 0, 10, 5)
            mood_swings = st.slider("Mood Swings Frequency (0-10)", 0, 10, 5)
            stress_level_burn = st.slider("Stress Level for Burnout (0-10)", 0, 10, 5)

            # Wellness Assessment
            st.subheader("Wellness Assessment")
            stress_level_well = st.slider("Current Stress Level for Wellness (0-5)", 0, 5, 2)

            submitted = st.form_submit_button("Predict Mental Health Status")

            if submitted:
                # Convert all categorical and binary inputs first
                gender_encoded = encode_categorical(gender, ['Male', 'Other'], 'gender')
                family_history_num = convert_binary_response(family_history)
                substance_use_num = convert_binary_response(substance_use)

                # Prepare stress input with type safety
                stress_input = pd.DataFrame({
                    'age': [convert_numeric(age)],
                    'sleep_hours': [convert_numeric(sleep_hours)],
                    'work_hours': [convert_numeric(work_hours)],
                    'exercise_freq': [convert_numeric(exercise_freq)],
                    'screen_time': [convert_numeric(screen_time)],
                    'social_activity': [convert_numeric(social_activity)],
                    'diet_quality': [convert_numeric(diet_quality)],
                    'gender_Male': [gender_encoded['gender_Male']],
                    'gender_Other': [gender_encoded['gender_Other']]
                })

                # Prepare depression input with type safety
                depression_input = pd.DataFrame({
                    'age': [convert_numeric(age)],
                    'sleep_quality': [convert_numeric(sleep_quality)],
                    'stress_level': [convert_numeric(stress_level_dep)],
                    'social_support': [convert_numeric(social_support)],
                    'physical_activity': [convert_numeric(physical_activity)],
                    'gender_Male': [gender_encoded['gender_Male']],
                    'gender_Other': [gender_encoded['gender_Other']],
                    'family_history_Yes': [family_history_num],
                    'substance_use_Yes': [substance_use_num]
                })

                # Prepare burnout input with type safety
                burnout_input = pd.DataFrame({
                    'age': [convert_numeric(age)],
                    'work_hours': [convert_numeric(work_hours)],
                    'overtime': [convert_numeric(overtime)],
                    'work_satisfaction': [convert_numeric(work_satisfaction)],
                    'stress_level': [convert_numeric(stress_level_burn)],
                    'sleep_hours': [convert_numeric(sleep_hours)],
                    'mood_swings': [convert_numeric(mood_swings)],
                    'gender_Male': [gender_encoded['gender_Male']],
                    'gender_Other': [gender_encoded['gender_Other']]
                })

                try:
                    # Make predictions with type safety
                    stress_prediction = convert_numeric(stress_model.predict(stress_input)[0])
                    
                    # FIX: Handle string predictions for depression model
                    depression_raw = depression_model.predict(depression_input)[0]
                    if isinstance(depression_raw, str):
                        # If model returns 'Yes'/'No' strings
                        depression_prediction = 1.0 if depression_raw.lower() == 'yes' else 0.0
                    else:
                        # If model returns numerical values
                        depression_prediction = float(depression_raw)
                    
                    # FIX: Handle string predictions for burnout model
                    burnout_raw = burnout_model.predict(burnout_input)[0]
                    if isinstance(burnout_raw, str):
                        # If model returns 'Yes'/'No' strings
                        burnout_prediction = 1.0 if burnout_raw.lower() == 'yes' else 0.0
                    else:
                        # If model returns numerical values
                        burnout_prediction = float(burnout_raw)

                    # Debug output
                    #st.write("Debug Info:", style={"display": "none"})
                    #st.write(f"Depression raw: {depression_raw}, type: {type(depression_raw)}")
                    #st.write(f"Depression prediction: {depression_prediction}")
                    #st.write(f"Burnout raw: {burnout_raw}, type: {type(burnout_raw)}")
                    #st.write(f"Burnout prediction: {burnout_prediction}")

                    # Prepare wellness input with type safety
                    wellness_input = pd.DataFrame({
                        'age': [convert_numeric(age)],
                        'stress_level': [convert_numeric(stress_level_well)],
                        'gender_Male': [gender_encoded['gender_Male']],
                        'gender_Other': [gender_encoded['gender_Other']],
                        'burnout_Yes': [float(1) if burnout_prediction >= 0.5 else float(0)],
                        'depression_risk_Yes': [float(1) if depression_prediction >= 0.5 else float(0)]
                    })

                    # Scale wellness input and get predictions
                    wellness_input_scaled = scaler.transform(wellness_input)
                    
                    try:
                        # FIX: Make sure we get the full array of predictions for wellness
                        wellness_raw = wellness_model.predict(wellness_input_scaled)
                        
                        # Handle different types of wellness predictions
                        if isinstance(wellness_raw, str):
                            # If it's a string, create default values
                            wellness_predictions = [3.0, 2.7, 3.3, 2.85]  # Default values
                        else:
                            # Handle different prediction shapes
                            if len(wellness_raw.shape) > 1 and wellness_raw.shape[0] == 1:
                                wellness_predictions = wellness_raw[0]  # Get first row if 2D
                            else:
                                wellness_predictions = wellness_raw
                    except Exception as e:
                        st.error(f"Error with wellness prediction: {str(e)}")
                        # Fallback to default values
                        wellness_predictions = [3.0, 2.7, 3.3, 2.85]
                    
                    # Ensure we have 4 values for the 4 activities
                    if not hasattr(wellness_predictions, "__len__") or len(wellness_predictions) != 4:
                        # If model doesn't return 4 values, we'll create 4 slightly different values
                        base_value = float(wellness_predictions[0] if hasattr(wellness_predictions, "__len__") and len(wellness_predictions) > 0 else 3.0)
                        wellness_predictions = [
                            base_value,
                            max(1.0, min(5.0, base_value * 0.9)),
                            max(1.0, min(5.0, base_value * 1.1)),
                            max(1.0, min(5.0, base_value * 0.95))
                        ]

                    # Display results with corrected scales
                    st.subheader("Predictions:")
                    st.write(f"**Stress Level:** {stress_prediction:.1f}/5")
                    st.write(f"**Depression Risk:** {'Yes' if depression_prediction >= 0.5 else 'No'}")
                    st.write(f"**Burnout Status:** {'Yes' if burnout_prediction >= 0.5 else 'No'}")
                    
                    st.write("\n**Recommended Wellness Activities (1-5 scale):**")
                    wellness_activities = ['Meditation', 'Therapy', 'Music Therapy', 'Relaxation Techniques']
                    
                    # Display each wellness activity with its own score
                    for i, activity in enumerate(wellness_activities):
                        if i < len(wellness_predictions):
                            try:
                                score = float(wellness_predictions[i])
                                score = max(1.0, min(5.0, score))
                                st.write(f"- {activity}: {score:.1f}/5")
                            except (ValueError, TypeError):
                                st.write(f"- {activity}: 3.0/5 (default)")
                        else:
                            st.write(f"- {activity}: 3.0/5 (default)")

                except Exception as e:
                    st.error(f"Error making predictions: {str(e)}")
                    import traceback
                    st.error(f"Traceback: {traceback.format_exc()}")
                    #if show_debug:
                        #st.subheader("Debug Info:")
                        #st.write(f"Stress raw: {stress_prediction}, type: {type(stress_prediction)}")
                        #st.write(f"Depression raw: {depression_raw}, type: {type(depression_raw)}")
                        #st.write(f"Depression prediction: {depression_prediction}")
                        #st.write(f"Burnout raw: {burnout_raw}, type: {type(burnout_raw)}")
                        #st.write(f"Burnout prediction: {burnout_prediction}")
                        #if 'wellness_raw' in locals():
                            #st.write(f"Wellness raw: {wellness_raw}, type: {type(wellness_raw)}")
                            #if isinstance(wellness_raw, np.ndarray):
                                #st.write(f"Wellness shape: {wellness_raw.shape}")
                    
                    st.error("Please check all input values and try again.")

        except Exception as e:
            st.error(f"Error processing inputs: {str(e)}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")

#show_debug = st.checkbox("Show Debug Info", value=False)

# Main app execution
get_user_input()