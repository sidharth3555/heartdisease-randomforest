import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model and scaler
rf_model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler_rf.pkl")

# Title
st.title("❤️ Heart Disease Prediction")
st.write("Random Forest model to predict heart disease.")

# Sidebar: User Input or File Upload
st.sidebar.header("Choose Input Method")
input_method = st.sidebar.radio("Select input type:", ["Manual Input", "Upload CSV"])

# Feature Input (Manual Entry)
if input_method == "Manual Input":
    st.sidebar.subheader("Enter Patient Details:")
    
    age = st.sidebar.number_input("Age", min_value=20, max_value=100, value=50)
    sex = st.sidebar.selectbox("Sex (1=Male, 0=Female)", [0, 1])
    cp = st.sidebar.slider("Chest Pain Type (0-3)", 0, 3, 1)
    trestbps = st.sidebar.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
    chol = st.sidebar.number_input("Cholesterol Level", min_value=100, max_value=600, value=200)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl (1=True, 0=False)", [0, 1])
    restecg = st.sidebar.slider("Resting ECG Results (0-2)", 0, 2, 1)
    thalach = st.sidebar.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
    exang = st.sidebar.selectbox("Exercise Induced Angina (1=Yes, 0=No)", [0, 1])
    oldpeak = st.sidebar.number_input("ST Depression", min_value=0.0, max_value=6.0, value=1.0, step=0.1)
    slope = st.sidebar.slider("Slope of ST Segment (0-2)", 0, 2, 1)
    ca = st.sidebar.slider("Number of Major Vessels (0-4)", 0, 4, 1)
    thal = st.sidebar.slider("Thalassemia (0-3)", 0, 3, 1)

    # Prepare input for prediction
    user_input = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    user_input_scaled = scaler.transform(user_input)

    if st.sidebar.button("Predict"):
        prediction = rf_model.predict(user_input_scaled)
        result = "Has Heart Disease" if prediction[0] == 1 else "No Heart Disease"
        st.success(f"Prediction: {result}")

# File Upload (CSV)
elif input_method == "Upload CSV":
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded Data:")
