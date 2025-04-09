import streamlit as st
import pandas as pd
import pickle

# Load model and encoder using pickle
with open("xgb_model.pkl", "rb") as model_file:
    xgb = pickle.load(model_file)

with open("ordinal_encoder.pkl", "rb") as encoder_file:
    Or = pickle.load(encoder_file)

st.set_page_config(page_title="Calories Burnt Prediction App", layout="centered")

st.title("🩺 Calories Burnt Prediction")
st.markdown("Provide your details below to predict your health status.")

# User inputs
gender = st.selectbox("Gender", ["male", "female"])
age = st.number_input("Age", min_value=1, max_value=100, value=21)
height = st.number_input("Height (cm)", min_value=0.0, max_value=250.0, value=170.0)
weight = st.number_input("Weight (kg)", min_value=0.0, max_value=200.0, value=67.0)
duration = st.number_input("Exercise Duration (minutes)", min_value=0.0, max_value=300.0, value=17.0)
heart_rate = st.number_input("Heart Rate (bpm)", min_value=0.0, max_value=200.0, value=95.0)
body_temp = st.number_input("Body Temperature (°C)", min_value=0.0, max_value=45.0, value=40.0)

# Prediction button
if st.button("Predict"):
    # Create input DataFrame
    input_data = pd.DataFrame([[gender, age, height, weight, duration, heart_rate, body_temp]],
                              columns=['Gender', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp'])

    # Encode gender
    input_data['Gender'] = Or.transform(input_data[['Gender']])

    # Make prediction
    prediction = xgb.predict(input_data)

    st.success(f"✅ Prediction Result: **{prediction[0]}**")
