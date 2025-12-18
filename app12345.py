import streamlit as st
import pickle
import numpy as np
import os

# Path to the trained model (NO comma)
model_path = '25RP20258.pkl'

st.title("🌱 Crop Yield Prediction Web App")
st.write("Predict crop yield based on temperature")

# Check if the model file exists
if os.path.exists(model_path):

    # Load the trained model
    model = pickle.load(open(model_path, 'rb'))

    # User input
    temperature = st.number_input(
        "Enter Average Temperature (°C)",
        min_value=0.0,
        step=0.1
    )

    if st.button("Predict"):
        prediction = model.predict(np.array([[temperature]]))
        st.success(
            f"Predicted Crop Yield: {prediction[0]:.2f} tons/hectare"
        )

else:
    st.error(
        f"Model file '{model_path}' not found. "
        "Please run your Jupyter Notebook to generate the model first."
    )
