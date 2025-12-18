"""
Progetto C – Assistente AI Multimodale (Testo + Voce + RAG)

Questo script è una versione compatta di un backend Flask che espone:
- /api/chat   : chat testuale con OpenAI + RAG semplice su un corpus locale
- /api/voice  : endpoint per ricevere audio, trascriverlo, rispondere e generare TTS

È pensato per essere affiancato da una semplice pagina HTML con:
- area chat
- bottone microfono (MediaRecorder)
- toggle "Usa RAG"

Prerequisiti:
- pip install flask openai sentence-transformers numpy
- OPENAI_API_KEY configurata
- modello sentence-transformers locale (es. ./hf/all-MiniLM-L6-v2)
"""

import os
import json
from flask import Flask, request, jsonify
from openai import OpenAI
import numpy as np
from sentence_transformers import SentenceTransformer
import tempfile
import uuid

app = Flask(__name__)
client = OpenAI()

RAG_MODEL_PATH = "./hf/all-MiniLM-L6-v2"
CORPUS_FILE = "manuale_tecnico.txt"
rag_model = None
rag_docs = []
rag_emb = None

SYSTEM_PROMPT = (
    "Sei un assistente AI del corso di Machine Learning. "
    "Rispondi sempre in italiano. "
    "Se il contesto RAG è presente, usalo come fonte principale."
)

def init_rag():
    global rag_model, rag_docs, rag_emb
    if not os.path.exists(CORPUS_FILE):
        print("manuale_tecnico.txt non trovato: RAG disabilitato.")
        return
    rag_model = SentenceTransformer(RAG_MODEL_PATH)
    with open(CORPUS_FILE, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
    rag_docs = chunks
    rag_emb = rag_model.encode(rag_docs)
    print(f"RAG inizializzato con {len(rag_docs)} chunk.")

def rag_search(query, top_k=3):
    if rag_model is None or not rag_docs:
        return ""
    q = rag_model.encode([query])[0]
    scores = []
    for e in rag_emb:
        s = float(np.dot(q, e) / ((np.linalg.norm(q)+1e-9)*(np.linalg.norm(e)+1e-9)))
        scores.append(s)
    idx = np.argsort(scores)[::-1][:top_k]
    selected = [rag_docs[i] for i in idx if scores[i] > 0]
    return "\n\n".join(selected)

@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json() or {}
    message = data.get("message", "")
    use_rag = bool(data.get("use_rag", False))
    if not message:
        return jsonify({"error": "Messaggio vuoto"}), 400

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if use_rag:
        ctx = rag_search(message)
        if ctx:
            messages.append(
                {
                    "role": "system",
                    "content": "Contesto dal manuale del corso:\n" + ctx
                }
            )
    messages.append({"role": "user", "content": message})

    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.5,
            max_tokens=400,
        )
        reply = res.choices[0].message.content
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"reply": reply})

@app.route("/api/voice", methods=["POST"])
def api_voice():
    audio_file = request.files.get("audio")
    use_rag = request.form.get("use_rag") == "1"
    if not audio_file:
        return jsonify({"error": "Nessun file audio"}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        temp_path = tmp.name
        audio_file.save(temp_path)

    try:
        with open(temp_path, "rb") as f:
            trans = client.audio.transcriptions.create(
                model="gpt-4o-mini-transcribe",
                file=f
            )
        text = trans.text
    except Exception as e:
        os.remove(temp_path)
        return jsonify({"error": f"Errore trascrizione: {e}"}), 500

    os.remove(temp_path)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if use_rag:
        ctx = rag_search(text)
        if ctx:
            messages.append(
                {
                    "role": "system",
                    "content": "Contesto dal manuale del corso:\n" + ctx
                }
            )
    messages.append({"role": "user", "content": text})

    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.5,
            max_tokens=400,
        )
        reply = res.choices[0].message.content
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    audio_url = None
    try:
        tts = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="alloy",
            input=reply,
            format="mp3",
        )
        audio_bytes = tts.read()
        tts_dir = os.path.join(app.static_folder, "tts")
        os.makedirs(tts_dir, exist_ok=True)
        fname = f"reply_{uuid.uuid4().hex}.mp3"
        fpath = os.path.join(tts_dir, fname)
        with open(fpath, "wb") as f:
            f.write(audio_bytes)
        audio_url = "/static/tts/" + fname
    except Exception as e:
        print("Errore TTS:", e)

    return jsonify({
        "transcript": text,
        "reply": reply,
        "audio_url": audio_url
    })

if __name__ == "__main__":
    os.makedirs(os.path.join("static", "tts"), exist_ok=True)
    init_rag()
    app.run(host="0.0.0.0", port=8000, debug=True)
