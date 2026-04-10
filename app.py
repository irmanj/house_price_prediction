import streamlit as st
import joblib
import pandas as pd
import numpy as np

model = joblib.load("model.pkl")

st.set_page_config(page_title="Prediksi Harga Rumah", layout="centered")

st.title("🏠 Prediksi Harga Rumah")
st.markdown("Masukkan data rumah untuk estimasi harga")

col1, col2 = st.columns(2)

with col1:
    luas = st.number_input("Luas Rumah (m2)", 20, 500, 50)

with col2: 
    kamar = st.number_input("Jumlah Kamar", 1, 10, 2)

if st.button("Prediksi Harga"):
    input_data = pd.DataFrame([[luas, kamar]], columns=["luas", "kamar"])
    harga = model.predict(input_data)[0]

    st.success(f"💰 Estimasi harga rumah: Rp {harga:.2f} juta")