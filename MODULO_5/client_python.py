import requests

payload = {
    "superficie": 100,
    "stanze": 4,
    "zona": 2
}

r = requests.post("http://localhost:5000/predict", json=payload)
print("Risposta API:", r.json())
