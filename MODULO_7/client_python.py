import requests

payload = {
    "superficie": 100,
    "stanze": 4,
    "zona": 2
}

r = requests.post("http://localhost:8000/ask", json=payload)
print("Risposta API:", r.json())
