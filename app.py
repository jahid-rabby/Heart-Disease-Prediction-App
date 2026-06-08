import streamlit as st
import pandas as pd
import joblib

# =========================
# Load Saved Files
# =========================
model = joblib.load("knn_model_heart.pkl")
scaler = joblib.load("scaler_heart.pkl")
expected_columns = joblib.load("feature_columns_heart.pkl")

# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="centered"
)

# =========================
# Title
# =========================
st.title("❤️ Heart Disease Prediction App")
st.subheader("Developed by Jahid Hossain Rabbi")

st.markdown(
    """
    Enter the patient's information below and click **Predict**
    to estimate the likelihood of heart disease.
    """
)

# =========================
# Sidebar
# =========================
st.sidebar.header("About")

st.sidebar.info(
    """
    This application predicts the likelihood of heart disease
    using a trained K-Nearest Neighbors (KNN) machine learning model.


    
    """
)

# =========================
# User Inputs
# =========================
age = st.number_input(
    "Age",
    min_value=0,
    max_value=120,
    value=40
)

sex = st.selectbox(
    "Sex",
    ["Male", "Female"]
)

chest_pain_type = st.selectbox(
    "Chest Pain Type",
    ["ATA", "NAP", "ASY", "TA"]
)

resting_bp = st.number_input(
    "Resting Blood Pressure (mmHg)",
    min_value=0,
    max_value=300,
    value=120
)

cholesterol = st.number_input(
    "Cholesterol",
    min_value=0,
    max_value=1000,
    value=200
)

fasting_blood_sugar = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dl",
    ["Yes", "No"]
)

rest_ecg = st.selectbox(
    "Resting ECG",
    [
        "Normal",
        "ST-T wave abnormality",
        "Left ventricular hypertrophy"
    ]
)

max_heart_rate = st.number_input(
    "Maximum Heart Rate",
    min_value=0,
    max_value=300,
    value=150
)

exercise_induced_angina = st.selectbox(
    "Exercise Induced Angina",
    ["Yes", "No"]
)

oldpeak = st.number_input(
    "Oldpeak",
    min_value=0.0,
    max_value=10.0,
    value=1.0
)

st_slope = st.selectbox(
    "ST Slope",
    ["Down", "Flat", "Up"]
)

# =========================
# Prediction Button
# =========================
if st.button("🔍 Predict"):

    row = {col: 0 for col in expected_columns}

    # Numerical Features
    row["Age"] = age
    row["RestingBP"] = resting_bp
    row["Cholesterol"] = cholesterol
    row["FastingBS"] = 1 if fasting_blood_sugar == "Yes" else 0
    row["MaxHR"] = max_heart_rate
    row["Oldpeak"] = oldpeak

    # Sex
    if "Sex_M" in row and sex == "Male":
        row["Sex_M"] = 1

    # Chest Pain Type
    if f"ChestPainType_{chest_pain_type}" in row:
        row[f"ChestPainType_{chest_pain_type}"] = 1

    # Resting ECG
    if rest_ecg == "Normal":
        if "RestingECG_Normal" in row:
            row["RestingECG_Normal"] = 1

    elif rest_ecg == "ST-T wave abnormality":
        if "RestingECG_ST" in row:
            row["RestingECG_ST"] = 1

    # Exercise Angina
    if exercise_induced_angina == "Yes":
        if "ExerciseAngina_Y" in row:
            row["ExerciseAngina_Y"] = 1

    # ST Slope
    if st_slope == "Flat":
        if "ST_Slope_Flat" in row:
            row["ST_Slope_Flat"] = 1

    elif st_slope == "Up":
        if "ST_Slope_Up" in row:
            row["ST_Slope_Up"] = 1

    # Convert to DataFrame
    input_df = pd.DataFrame([row])

    try:
        # Scale Input
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = model.predict(input_scaled)[0]

        # Probability
        probability = model.predict_proba(input_scaled)[0][1]

        st.markdown("---")

        if prediction == 1:
            st.error(
                f"⚠️ High Risk of Heart Disease\n\n"
                f"Predicted Risk Probability: {probability:.2%}"
            )
        else:
            st.success(
                f"✅ Low Risk of Heart Disease\n\n"
                f"Predicted Risk Probability: {probability:.2%}"
            )

    except Exception as e:
        st.error(f"Error during prediction:\n{e}")