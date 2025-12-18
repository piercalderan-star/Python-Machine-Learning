# app.py
import os
from flask import Flask, render_template, request, jsonify
from openai import OpenAI

app = Flask(__name__)

# Usa la chiave da variabile d'ambiente OPENAI_API_KEY
client = OpenAI()

SYSTEM_PROMPT = (
    "Sei un assistente AI gentile e competente. "
    "Rispondi in italiano, in modo chiaro e conciso. "
    "Se l'utente chiede codice, usa Python quando possibile."
)

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/api/chat", methods=["POST"])
def chat_api():
    data = request.get_json() or {}
    messages = data.get("messages", [])
    user_message = data.get("message", "")

    # Se non arriva la history dal frontend, creiamo una conversazione minima
    if not messages and user_message:
        messages = [{"role": "user", "content": user_message}]

    # Inseriamo sempre il system prompt in testa
    full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=full_messages,
            temperature=0.7,
            max_tokens=300,
        )
        reply = res.choices[0].message.content
        return jsonify({"reply": reply})
    except Exception as e:
        # In caso di errore API, restituiamo un messaggio comprensibile al frontend
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Per sviluppo locale
    app.run(host="0.0.0.0", port=5000, debug=True)
