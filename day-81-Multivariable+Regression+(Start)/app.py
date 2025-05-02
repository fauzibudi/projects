# app.py

import streamlit as st
import pandas as pd
import joblib
import numpy as np 

# Load model
model = joblib.load('model.pkl')

st.title("Prediksi Harga Rumah Boston")

# Ambil nama kolom dari data
feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 
                 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']


# Buat input untuk setiap fitur
user_input = {}
for feature in feature_names:
    user_input[feature] = st.number_input(f"Masukkan nilai untuk {feature}", value=0.0)

# Jika tombol ditekan
if st.button("Prediksi"):
    input_df = pd.DataFrame([user_input])
    log_pred = model.predict(input_df)[0]
    prediction = np.exp(log_pred)
    st.success(f"Prediksi harga rumah: ${prediction * 1000:,.2f}")
