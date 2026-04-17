from fastapi import FastAPI
import pickle 
import numpy as np
import pandas as pd
import os

port = int(os.environ.get("PORT", 8000))

app = FastAPI()

# load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.get("/")
def home():
    return {"message": "House Price API"}

@app.post("/predict")
def predict(luas_rumah: float, jumlah_kamar: int, lokasi: str):
    
    # validasi lokasi
    if lokasi not in ["Jakarta", "Bandung", "Depok", "Bogor"]:
        return {"error": "Lokasi tidak valid"}

    data = pd.DataFrame([{
        "luas_rumah": luas_rumah,
        "jumlah_kamar": jumlah_kamar,
        "lokasi": lokasi
    }])

    pred = model.predict(data)[0]

    return {
        "input": {
            "luas_rumah": luas_rumah,
            "jumlah_kamar": jumlah_kamar,
            "lokasi": lokasi
        },
        "prediction": f"Rp {int(pred):,}"
    }

@app.get("/health")
def health():
    return {"status": "ok"}