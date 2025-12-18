import streamlit as st
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open('25RP20258.pkl', 'rb'))

st.title("Crop Yield Prediction Web App")
st.write("Predict crop yield based on temperature")

temperature = st.number_input(
    "Enter Average Temperature (°C)",
    min_value=0.0,
    step=0.1)
if st.button("Predict"):
    prediction = model.predict(np.array([[temperature]]))
    st.success(f"Predicted Crop Yield: {prediction[0]:.2f} tons/hectare")


