# api_flask.py
'''
esempio di comando post
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"superficie\":85, \"stanze\":3, \"zona\":1}"
 oppure lanciare client_python.py
 creare un client node-red http post http://localhost:8000/predict
 header Content-Type: application/json
 inject
{ superficie = data["superficie"]
    stanze = data["stanze"]
    zona = data["zona"]
  }
'''

from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Caricamento modello all’avvio
model = joblib.load("modello_case.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    superficie = data["superficie"]
    stanze = data["stanze"]
    zona = data["zona"]

    X = np.array([[superficie, stanze, zona]])
    pred = model.predict(X)[0]

    return jsonify({
        "prezzo_stimato": round(float(pred), 2)
    })

@app.route("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
