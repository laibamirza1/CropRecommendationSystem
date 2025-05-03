import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load model, encoder, and scaler
model = joblib.load("crop_model.pkl")
encoder = joblib.load("label_encoder.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Crop Recommendation", layout="centered")
st.title("🌱 Crop Recommendation System")
st.markdown("Provide soil and environmental conditions to get the best crop suggestion.")

# Input form
with st.form("input_form"):
    N = st.number_input("Nitrogen (N)", 0, 140, step=1)
    P = st.number_input("Phosphorus (P)", 0, 140, step=1)
    K = st.number_input("Potassium (K)", 0, 140, step=1)
    temperature = st.slider("Temperature (°C)", 0.0, 50.0, 25.0)
    humidity = st.slider("Humidity (%)", 0.0, 100.0, 50.0)
    ph = st.slider("pH Level", 3.0, 10.0, 6.5)
    rainfall = st.slider("Rainfall (mm)", 0.0, 300.0, 100.0)
    submitted = st.form_submit_button("Recommend Crop")

# Define the crop names in the same order as the encoder's output
crop_names = ['rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas', 'mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate',
              'banana', 'mango', 'grapes', 'watermelon', 'muskmelon', 'apple', 'orange', 'papaya', 'coconut', 'cotton', 'jute', 'coffee']

if submitted:
    # Prepare the input data
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

    # Scale the input data using the same scaler that was used during training
    input_data_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data_scaled)[0]  # This is a number (integer)

    # Manually map prediction number to crop name
    try:
        # Print the predicted number
        st.write("Raw prediction (index):", prediction)
        
        # Map prediction to crop name
        if 0 <= prediction < len(crop_names):
            crop_name = crop_names[prediction]
            st.success(f"🌾 Recommended Crop: **{crop_name.upper()}**")
        else:
            st.error("Prediction out of bounds, check your model.")
    except Exception as e:
        st.error(f"An error occurred while decoding prediction: {e}")
        st.write("Raw prediction:", prediction)

# Add custom styling from the style.css file
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
