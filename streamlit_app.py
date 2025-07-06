# streamlit_app.py

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime

# —————————————————————————————
# DATABASE SETUP
# —————————————————————————————
conn = sqlite3.connect("predictions.db", check_same_thread=False)
c = conn.cursor()
c.execute("""
    CREATE TABLE IF NOT EXISTS churn_predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        gender TEXT,
        SeniorCitizen INTEGER,
        Partner TEXT,
        Dependents TEXT,
        tenure INTEGER,
        PhoneService TEXT,
        MultipleLines TEXT,
        InternetService TEXT,
        OnlineSecurity TEXT,
        OnlineBackup TEXT,
        DeviceProtection TEXT,
        TechSupport TEXT,
        StreamingTV TEXT,
        StreamingMovies TEXT,
        Contract TEXT,
        PaperlessBilling TEXT,
        PaymentMethod TEXT,
        MonthlyCharges REAL,
        TotalCharges REAL,
        tenure_group TEXT,
        prediction INTEGER,
        probability REAL
    )
""")
conn.commit()

# (Removed duplicate Predict Churn button and block that used undefined variables)


# Load model
pipeline = joblib.load("models/churn_pipeline.pkl")

st.title("📊 Customer Churn Prediction App")
st.write("Enter customer details to predict churn likelihood.")

# User Inputs
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Has Partner", ["Yes", "No"])
dependents = st.selectbox("Has Dependents", ["Yes", "No"])
tenure = st.slider("Tenure (months)", 0, 72, 12)
phone_service = st.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
monthly_charges = st.slider("Monthly Charges", 0.0, 150.0, 70.0)
total_charges = st.slider("Total Charges", 0.0, 10000.0, 2500.0)

# Create a DataFrame from inputs
input_dict = {
    'gender': gender,
    'SeniorCitizen': senior,
    'Partner': partner,
    'Dependents': dependents,
    'tenure': tenure,
    'PhoneService': phone_service,
    'MultipleLines': multiple_lines,
    'InternetService': internet_service,
    'OnlineSecurity': online_security,
    'OnlineBackup': online_backup,
    'DeviceProtection': device_protection,
    'TechSupport': tech_support,
    'StreamingTV': streaming_tv,
    'StreamingMovies': streaming_movies,
    'Contract': contract,
    'PaperlessBilling': paperless,
    'PaymentMethod': payment_method,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges
}
input_df = pd.DataFrame([input_dict])

# Add tenure_group feature
def get_tenure_group(tenure):
    if tenure <= 12:
        return '0-12'
    elif tenure <= 24:
        return '13-24'
    elif tenure <= 36:
        return '25-36'
    elif tenure <= 48:
        return '37-48'
    elif tenure <= 60:
        return '49-60'
    else:
        return '61-72'

input_df['tenure_group'] = input_df['tenure'].apply(get_tenure_group)

if st.checkbox("Show Prediction History"):
    df_hist = pd.read_sql_query("SELECT * FROM churn_predictions ORDER BY id DESC LIMIT 20", conn)
    st.dataframe(df_hist)

# Predict
if st.button("Predict Churn"):
    # … your existing predict code …
    prediction = pipeline.predict(input_df)[0]
    prob       = pipeline.predict_proba(input_df)[0][1]

    # Display result
    if prediction == 1:
        st.error(f"⚠️ Likely to Churn ({prob*100:.2f}%)")
    else:
        st.success(f"✅ Not Likely to Churn ({(1-prob)*100:.2f}%)")

    # —————————————————————————————
    # STORE IN DATABASE
    # —————————————————————————————
    ts = datetime.now().isoformat(timespec='seconds')
    # Build a tuple of all fields in order
    record = (
        ts,
        gender, senior, partner, dependents, tenure,
        phone_service, multiple_lines, internet_service,
        online_security, online_backup, device_protection,
        tech_support, streaming_tv, streaming_movies,
        contract, paperless, payment_method,
        monthly_charges, total_charges,
        input_df['tenure_group'][0],
        int(prediction), float(prob)
    )
    c.execute("""
        INSERT INTO churn_predictions (
            timestamp, gender, SeniorCitizen, Partner, Dependents, tenure,
            PhoneService, MultipleLines, InternetService, OnlineSecurity,
            OnlineBackup, DeviceProtection, TechSupport, StreamingTV,
            StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
            MonthlyCharges, TotalCharges, tenure_group, prediction, probability
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, record)
    conn.commit()
