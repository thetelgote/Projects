# app.py
import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("model.pkl")

# App title
st.title("🏡 House Price Prediction App")

# Input form
area = st.number_input("Enter Area (sq ft):", min_value=500, max_value=10000)
bedrooms = st.number_input("Enter Bedrooms:", min_value=1, max_value=10)
bathrooms = st.number_input("Enter Bathrooms:", min_value=1, max_value=10)

# Predict button
if st.button("Predict"):
    input_data = pd.DataFrame([[area, bedrooms, bathrooms]], columns=["area", "bedrooms", "bathrooms"])
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated Price: ₹ {prediction:.2f} Lakhs")
