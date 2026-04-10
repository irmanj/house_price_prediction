import streamlit as st
import joblib
import pandas as pd
import numpy as np

model = joblib.load("model.pkl")

st.title("🏠 Prediksi Harga Rumah")

luas = st.slider("Luas Rumah", 20, 200)
kamar = st.slider("Jumlah Kamar", 1, 5)

# luas = float(input("Masukkan luas rumah: "))
# kamar = int(input("Jumlah kamar: "))

if st.button("Prediksi"):
    input_data = pd.DataFrame([[luas, kamar]], columns=["luas", "kamar"])
    # input_data = np.array([[luas, kamar]])
    harga = model.predict(input_data)[0]

    st.success(f"Estimasi harga rumah: {harga:.2f} juta")