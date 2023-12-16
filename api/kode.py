from fastapi import FastAPI
from pydantic import BaseModel
from keras.preprocessing.text import Tokenizer
from pathlib import Path
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import numpy as np
import pickle

app = FastAPI()

BASE_DIR = Path(__file__).resolve(strict=True).parent
kelas = ["Wanita", "Pria"]
model = load_model(f"../model/model_tyok.h5")
with open("../model/tokenizer.pkl", "rb") as f:
    new_token = pickle.load(f)

class InputModel(BaseModel):
    nama:str

@app.post("/predict")
def predict(nama: InputModel):
    data = nama.nama
    seq = new_token.texts_to_sequences([data])
    padding = pad_sequences(seq, maxlen=20)
    prediksi = model.predict(padding)
    isKelas = 1 if prediksi[0] > 0.56 else 0 
    return {
        "hasil": kelas[isKelas]
    }
    # print(prediksi)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
