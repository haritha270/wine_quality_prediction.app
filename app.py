import streamlit as st
import pandas as pd
import pickle
import numpy as np

# -----------------------------
# Load Model & Scaler
# -----------------------------
@st.cache_resource
def load_model():
    model = pickle.load(open("finalized_model_LogReg.sav", "rb"))
    scaler = pickle.load(open("scaler_model.sav", "rb"))
    return model, scaler

model, scaler = load_model()

# -----------------------------
# App Title
# -----------------------------
st.set_page_config(page_title="Wine Quality Prediction", layout="centered")
st.title("🍷 Wine Quality Prediction")
st.markdown("**Mini Project using Machine Learning (Logistic Regression)**")

st.divider()

# -----------------------------
# User Inputs
# -----------------------------
st.subheader("Enter Wine Chemical Properties")

fixed_acidity = st.number_input("Fixed Acidity", value=7.4)
volatile_acidity = st.number_input("Volatile Acidity", value=0.70)
citric_acid = st.number_input("Citric Acid", value=0.00)
residual_sugar = st.number_input("Residual Sugar", value=0.65)
chlorides = st.number_input("Chlorides", value=0.08)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", value=15.0)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", value=46.0)
density = st.number_input("Density", value=0.997)
pH = st.number_input("pH", value=3.51)
sulphates = st.number_input("Sulphates", value=0.56)
alcohol = st.number_input("Alcohol", value=9.4)

# -----------------------------
# Create Input DataFrame
# -----------------------------
input_data = pd.DataFrame([{
    'fixed acidity': fixed_acidity,
    'volatile acidity': volatile_acidity,
    'citric acid': citric_acid,
    'residual sugar': residual_sugar,
    'chlorides': chlorides,
    'free sulfur dioxide': free_sulfur_dioxide,
    'total sulfur dioxide': total_sulfur_dioxide,
    'density': density,
    'pH': pH,
    'sulphates': sulphates,
    'alcohol': alcohol
}])

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict Wine Quality"):
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]

    st.success(f"🍾 Predicted Wine Quality Class: **{prediction}**")

st.divider()
st.caption("Mini Project | Wine Quality Prediction using Machine Learning")
