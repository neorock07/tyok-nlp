from fastapi import FastAPI
from pydantic import BaseModel
from keras.preprocessing.text import Tokenizer
from pathlib import Path
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import numpy as np
import pickle

"""
Definisikan objek fast-api di sini.
"""
app = FastAPI()

"""
Load model dan tokenizer, pastikan lokasi folder sesuai.
definisikan juga kelas label yang dipakai saat pelatihan model.
"""
BASE_DIR = Path(__file__).resolve(strict=True).parent
kelas = ["Wanita", "Pria"]
model = load_model(f"app/model/model_tyok.h5")
with open("app/model/tokenizer.pkl", "rb") as f:
    new_token = pickle.load(f)

"""
Buat class input dengan pydantic agar memudahkan
penanganan error.
di sini kita pakai 1 inputan untuk `nama`.
"""
class InputModel(BaseModel):
    nama:str

"""
buat routing API dengan membuat function sesuai nama route kita.
misal kita mau buat rute : https://localhost:8000/api/predict, 
nah kita tambahkan annotation seusai metode request nya (POST, GET, PUT).
"""
@app.post("/predict")
def predict(nama: InputModel):
    """
    kita pre-processing data string menjadi vektor embedding, 
    sesuai saat kita melatih model NLP.
    """
    data = nama.nama
    seq = new_token.texts_to_sequences([data])
    padding = pad_sequences(seq, maxlen=20)
    """
    panggil function predict untuk melakukan prediksi terhadap
    data input yang sudah kita pre-process.
    """
    prediksi = model.predict(padding)
    isKelas = 1 if prediksi[0] > 0.56 else 0 
    """
    result dari API kita dalam bentuk JSON, maka dari itu
    kita akan return kan dalam bentuk map ({'key' : 'value'}).
    """
    return {
        "hasil": kelas[isKelas]
    }
    # print(prediksi)

"""
kode untuk mengatur fast-api untuk jalan di port 8000;
"""
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
