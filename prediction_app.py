import streamlit as st
import pandas as pd
import joblib

# Load model, scaler and columns
model = joblib.load("brest_cancer.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🎗️"
)

st.title(" Breast Cancer Prediction")
st.write("Enter tumor characteristics and click Predict.")


# Mean Features

texture_mean = st.number_input("Texture Mean", value=10.38)
smoothness_mean = st.number_input("Smoothness Mean", value=0.11840)
compactness_mean = st.number_input("Compactness Mean", value=0.27760)
concave_points_mean = st.number_input("Concave Points Mean", value=0.14710)
symmetry_mean = st.number_input("Symmetry Mean", value=0.24190)
fractal_dimension_mean = st.number_input("Fractal Dimension Mean", value=0.07871)

# =========================
# Standard Error Features
# =========================

texture_se = st.number_input("Texture SE", value=0.9053)
area_se = st.number_input("Area SE", value=153.40)
smoothness_se = st.number_input("Smoothness SE", value=0.006399)
compactness_se = st.number_input("Compactness SE", value=0.04904)
concavity_se = st.number_input("Concavity SE", value=0.05373)
concave_points_se = st.number_input("Concave Points SE", value=0.01587)
symmetry_se = st.number_input("Symmetry SE", value=0.03003)
fractal_dimension_se = st.number_input("Fractal Dimension SE", value=0.006193)

# Worst Features

texture_worst = st.number_input("Texture Worst", value=17.33)
area_worst = st.number_input("Area Worst", value=2019.0)
smoothness_worst = st.number_input("Smoothness Worst", value=0.16220)
compactness_worst = st.number_input("Compactness Worst", value=0.66560)
concavity_worst = st.number_input("Concavity Worst", value=0.71190)
concave_points_worst = st.number_input("Concave Points Worst", value=0.26540)
symmetry_worst = st.number_input("Symmetry Worst", value=0.46010)
fractal_dimension_worst = st.number_input("Fractal Dimension Worst", value=0.11890)

# Prediction

if st.button("Predict"):

    input_data = pd.DataFrame({
        'texture_mean': [texture_mean],
        'smoothness_mean': [smoothness_mean],
        'compactness_mean': [compactness_mean],
        'concave points_mean': [concave_points_mean],
        'symmetry_mean': [symmetry_mean],
        'fractal_dimension_mean': [fractal_dimension_mean],

        'texture_se': [texture_se],
        'area_se': [area_se],
        'smoothness_se': [smoothness_se],
        'compactness_se': [compactness_se],
        'concavity_se': [concavity_se],
        'concave points_se': [concave_points_se],
        'symmetry_se': [symmetry_se],
        'fractal_dimension_se': [fractal_dimension_se],

        'texture_worst': [texture_worst],
        'area_worst': [area_worst],
        'smoothness_worst': [smoothness_worst],
        'compactness_worst': [compactness_worst],
        'concavity_worst': [concavity_worst],
        'concave points_worst': [concave_points_worst],
        'symmetry_worst': [symmetry_worst],
        'fractal_dimension_worst': [fractal_dimension_worst]
    })

    # Match exact training columns
    input_data = input_data.reindex(
        columns=expected_columns,
        fill_value=0
    )

    # Scale data
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("⚠️ Malignant Tumor Detected")
    else:
        st.success("✅ Benign Tumor Detected")

