# api_fastapi.py
'''
 Uvicorn è un server web ASGI (Asynchronous Server Gateway Interface),
 veloce e leggero, progettato per eseguire applicazioni web Python asincrone.
 Funziona come un ponte tra il codice della tua applicazione
 (come quelle create con FastAPI o Starlette) e Internet,
 gestendo le richieste di rete in ingresso.
 Swagger è un insieme di strumenti basati sulla specifica OpenAPI,
 uno standard indipendente dal linguaggio per descrivere le API REST.
 L'interfaccia utente di Swagger (Swagger UI)
 fornisce una rappresentazione interattiva e visuale di questa descrizione.

 Esecuzione pip install uvicorn
 uvicorn api_fastapi:app --reload --port 8000

 uvicorn 02_api_fastapi:app --reload --port 8000
 
 Swagger disponibile qui:
 http://localhost:8000/docs
'''

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Modello input
class House(BaseModel):
    superficie: float
    stanze: int
    zona: int

app = FastAPI(title="House Price API")

model = joblib.load("modello_case.pkl")

@app.post("/predict")
def predict(data: House):
    X = np.array([[data.superficie, data.stanze, data.zona]])
    pred = model.predict(X)[0]
    return {"prezzo_stimato": round(float(pred), 2)}

@app.get("/health")
def health():
    return {"status": "ok"}


