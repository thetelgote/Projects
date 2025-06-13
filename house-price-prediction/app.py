# app.py

import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("model.pkl")

st.title("🏠 House Price Prediction App")

# User inputs
area = st.number_input("Area (in sq ft):", min_value=500, max_value=10000)
bedrooms = st.number_input("Number of bedrooms:", min_value=1, max_value=10)
bathrooms = st.number_input("Number of bathrooms:", min_value=1, max_value=10)

if st.button("Predict"):
    input_data = pd.DataFrame([[area, bedrooms, bathrooms]], columns=["Area", "Bedrooms", "Bathrooms"])
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Price: ₹ {prediction:.2f}")
