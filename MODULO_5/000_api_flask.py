import joblib
import pandas as pd
from flask import Flask, request, jsonify
import json
import os

# Assicurati che il modello esista prima di avviare l'app
MODEL_PATH = 'modello_prezzi_casa_rf.pkl'

if not os.path.exists(MODEL_PATH):
    print(f"Errore: File modello non trovato in {MODEL_PATH}")
    print("Esegui prima lo script precedente (Random Forest) per crearlo.")
    exit()

# Carica il modello addestrato una volta all'avvio dell'applicazione Flask
model = joblib.load(MODEL_PATH)
app = Flask(__name__)

@app.route('/')
def home():
    return "API di Previsione Prezzi Case Attiva. Usa l'endpoint /predict con una richiesta POST."

@app.route('/predict', methods=['POST'])
def predict():
    # Ottieni i dati inviati come JSON
    data = request.get_json(force=True)
    
    # Valida i campi richiesti
    required_fields = ['metri_quadrati', 'stanze', 'zona']
    if not all(field in data for field in required_fields):
        return jsonify({"error": "Campi mancanti. Richiesti: metri_quadrati, stanze, zona"}), 400

    try:
        # Prepara i dati in un formato compatibile con il modello (DataFrame)
        # Nota: i nomi delle colonne devono corrispondere a quelli usati in addestramento
        input_data = pd.DataFrame([[
            data['metri_quadrati'],
            data['stanze'],
            data['zona']
        ]], columns=['Metri Quadrati', 'Stanze', 'Zona'])
        
        # Esegui la previsione
        prediction = model.predict(input_data)
        
        # Formatta la risposta JSON
        response = {
            'prezzo_previsto': float(prediction[0]),
            'metri_quadrati': data['metri_quadrati'],
            'stanze': data['stanze'],
            'zona': data['zona']
        }
        
        return jsonify(response)

    except ValueError as e:
        return jsonify({"error": f"Errore di tipo dati. Assicurati che i valori siano numerici. Dettaglio: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Errore interno del server: {str(e)}"}), 500

if __name__ == '__main__':
    # Esegui l'app Flask. Usa debug=True per il testing locale.
    # L'app sarà disponibile su 127.0.0.1
    app.run(debug=True, port=5000)
