# app.py
import os
import json
import sqlite3
from datetime import datetime

from flask import Flask, render_template, request, jsonify
from openai import OpenAI

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

DB_PATH = "chat_history.db"
RAG_INDEX_PATH = "rag_index.json"

app = Flask(__name__)
client = OpenAI()

SYSTEM_PROMPT = (
    "Sei un assistente AI gentile e competente. "
    "Rispondi sempre in italiano, in modo chiaro e sintetico. "
    "Se l'utente chiede codice, preferisci esempi in Python."
)

# --- RAG setup (senza chromadb) ---
RAG_ENABLED = True
rag_model = None
rag_docs = []  # lista di dict: { "text": str, "source": "faq"/"manuale", "embedding": np.array }


def init_rag_model():
    global rag_model
    if rag_model is None:
        try:
            rag_model = SentenceTransformer("./hf/all-MiniLM-L6-v2")
        except Exception as e:
            print("ATTENZIONE: impossibile caricare SentenceTransformer per RAG:", e)
            return False
    return True


def save_rag_index():
    """Salva rag_docs in JSON (embedding come lista di float)."""
    if not rag_docs:
        return
    serializable = []
    for d in rag_docs:
        serializable.append({
            "text": d["text"],
            "source": d["source"],
            "embedding": d["embedding"].tolist()
        })
    with open(RAG_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False)


def load_rag_index():
    """Carica l'indice se esiste già."""
    global rag_docs
    if not os.path.exists(RAG_INDEX_PATH):
        return False
    try:
        with open(RAG_INDEX_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        rag_docs = []
        for d in data:
            rag_docs.append({
                "text": d["text"],
                "source": d.get("source", "manuale"),
                "embedding": np.array(d["embedding"], dtype=np.float32)
            })
        print(f"Caricato indice RAG da {RAG_INDEX_PATH} con {len(rag_docs)} documenti.")
        return True
    except Exception as e:
        print("Errore nel caricamento dell'indice RAG:", e)
        return False


def build_rag_index():
    """Costruisce l'indice RAG da faq_prodotti.csv e manuale_tecnico.txt."""
    global rag_docs

    if not init_rag_model():
        return

    rag_docs = []

    # FAQ
    if os.path.exists("faq_prodotti.csv"):
        try:
            # Primo tentativo: UTF-8
            try:
                df = pd.read_csv("faq_prodotti.csv", encoding="utf-8")
            except UnicodeDecodeError:
                # Fallback: latin-1 (tipico Windows / Excel in italiano)
                df = pd.read_csv("faq_prodotti.csv", encoding="latin-1")

            if "domanda" in df.columns:
                domande = df["domanda"].fillna("").tolist()
                emb = rag_model.encode(domande)
                for txt, e in zip(domande, emb):
                    rag_docs.append({
                        "text": txt,
                        "source": "faq",
                        "embedding": e.astype(np.float32)
                    })
                print(f"Aggiunte {len(domande)} FAQ all'indice RAG.")
            else:
                print("Colonna 'domanda' non trovata in faq_prodotti.csv")
        except Exception as e:
            print("Errore nel caricamento faq_prodotti.csv:", e)

    # Manuale
    if os.path.exists("manuale_tecnico.txt"):
        try:
            # Primo tentativo: UTF-8
            try:
                with open("manuale_tecnico.txt", "r", encoding="utf-8") as f:
                    text = f.read()
            except UnicodeDecodeError:
                # Fallback: latin-1
                with open("manuale_tecnico.txt", "r", encoding="latin-1") as f:
                    text = f.read()

            chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
            emb = rag_model.encode(chunks)
            for txt, e in zip(chunks, emb):
                rag_docs.append({
                    "text": txt,
                    "source": "manuale",
                    "embedding": e.astype(np.float32)
                })
            print(f"Aggiunti {len(chunks)} chunk del manuale all'indice RAG.")
        except Exception as e:
            print("Errore nel caricamento manuale_tecnico.txt:", e)

    if rag_docs:
        print(f"Indice RAG costruito con totale {len(rag_docs)} documenti.")
        save_rag_index()
    else:
        print("Nessun documento per RAG trovato.")



##def build_rag_index():
##    """Costruisce l'indice RAG da faq_prodotti.csv e manuale_tecnico.txt."""
##    global rag_docs
##
##    if not init_rag_model():
##        return
##
##    rag_docs = []
##
##    # FAQ
##    if os.path.exists("faq_prodotti.csv"):
##        try:
##            df = pd.read_csv("faq_prodotti.csv")
##            if "domanda" in df.columns:
##                domande = df["domanda"].fillna("").tolist()
##                emb = rag_model.encode(domande)
##                for txt, e in zip(domande, emb):
##                    rag_docs.append({
##                        "text": txt,
##                        "source": "faq",
##                        "embedding": e.astype(np.float32)
##                    })
##                print(f"Aggiunte {len(domande)} FAQ all'indice RAG.")
##        except Exception as e:
##            print("Errore nel caricamento faq_prodotti.csv:", e)
##
##    # Manuale
##    if os.path.exists("manuale_tecnico.txt"):
##        try:
##            with open("manuale_tecnico.txt", "r", encoding="utf-8") as f:
##                text = f.read()
##            chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
##            emb = rag_model.encode(chunks)
##            for txt, e in zip(chunks, emb):
##                rag_docs.append({
##                    "text": txt,
##                    "source": "manuale",
##                    "embedding": e.astype(np.float32)
##                })
##            print(f"Aggiunti {len(chunks)} chunk del manuale all'indice RAG.")
##        except Exception as e:
##            print("Errore nel caricamento manuale_tecnico.txt:", e)
##
##    if rag_docs:
##        print(f"Indice RAG costruito con totale {len(rag_docs)} documenti.")
##        save_rag_index()
##    else:
##        print("Nessun documento per RAG trovato.")
##       


def ensure_rag_index():
    """Garantisce che l'indice RAG sia pronto (caricato o costruito)."""
    global RAG_ENABLED
    if not RAG_ENABLED:
        return
    if rag_docs:
        return
    # Primo tentativo: caricare da file
    if not load_rag_index():
        # Secondo tentativo: costruirlo
        build_rag_index()
    if not rag_docs:
        print("RAG disabilitato: nessun documento disponibile.")
        RAG_ENABLED = False


def get_rag_context(query: str, top_k: int = 3) -> str:
    """Ritorna un contesto testuale combinando i documenti più simili (FAQ + manuale)."""
    if not RAG_ENABLED:
        return ""

    ensure_rag_index()
    if not rag_docs or not init_rag_model():
        return ""

    # Embedding query
    q_emb = rag_model.encode([query])[0].astype(np.float32)

    # Similarità coseno: cos_sim(a,b) = a·b / (||a||*||b||)
    q_norm = np.linalg.norm(q_emb) + 1e-9

    scores = []
    for d in rag_docs:
        doc_emb = d["embedding"]
        denom = (np.linalg.norm(doc_emb) + 1e-9) * q_norm
        sim = float(np.dot(q_emb, doc_emb) / denom)
        scores.append(sim)

    # ordiniamo i doc per similarità
    idx_sorted = np.argsort(scores)[::-1]  # descending
    top_idx = idx_sorted[:top_k]

    selected_docs = [rag_docs[i]["text"] for i in top_idx if scores[i] > 0.0]

    if not selected_docs:
        return ""

    context = "\n\n".join(selected_docs)
    return context


# --- DB CHAT (SQLite) ---

def init_db():
    """Crea le tabelle SQLite se non esistono."""
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            created_at TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER,
            role TEXT,
            content TEXT,
            created_at TEXT,
            FOREIGN KEY(conversation_id) REFERENCES conversations(id)
        )
        """
    )
    con.commit()
    con.close()


def get_db_connection():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


# --- ROUTES ---

@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/api/conversations", methods=["GET"])
def list_conversations():
    con = get_db_connection()
    cur = con.cursor()
    cur.execute(
        "SELECT id, title, created_at FROM conversations ORDER BY id DESC"
    )
    rows = cur.fetchall()
    con.close()
    convs = [
        {
            "id": r["id"],
            "title": r["title"],
            "created_at": r["created_at"],
        }
        for r in rows
    ]
    return jsonify(convs)


@app.route("/api/conversations", methods=["POST"])
def create_conversation():
    data = request.get_json() or {}
    title = data.get("title") or "Nuova conversazione"
    now = datetime.utcnow().isoformat(timespec="seconds")

    con = get_db_connection()
    cur = con.cursor()
    cur.execute(
        "INSERT INTO conversations (title, created_at) VALUES (?, ?)",
        (title, now),
    )
    conv_id = cur.lastrowid
    con.commit()
    con.close()
    return jsonify({"id": conv_id, "title": title, "created_at": now})


@app.route("/api/conversations/<int:conv_id>", methods=["GET"])
def get_conversation(conv_id):
    con = get_db_connection()
    cur = con.cursor()
    cur.execute(
        """
        SELECT id, title, created_at
        FROM conversations
        WHERE id = ?
        """,
        (conv_id,),
    )
    conv = cur.fetchone()
    if not conv:
        con.close()
        return jsonify({"error": "Conversazione non trovata"}), 404

    cur.execute(
        """
        SELECT role, content, created_at
        FROM messages
        WHERE conversation_id = ?
        ORDER BY id ASC
        """,
        (conv_id,),
    )
    rows = cur.fetchall()
    con.close()

    messages = [
        {
            "role": r["role"],
            "content": r["content"],
            "created_at": r["created_at"],
        }
        for r in rows
    ]

    return jsonify(
        {
            "id": conv["id"],
            "title": conv["title"],
            "created_at": conv["created_at"],
            "messages": messages,
        }
    )


@app.route("/api/chat", methods=["POST"])
def chat_api():
    data = request.get_json() or {}
    messages = data.get("messages", [])
    user_message = data.get("message", "")
    conv_id = data.get("conversation_id")
    use_rag = bool(data.get("use_rag", False))

    if not user_message:
        return jsonify({"error": "Messaggio vuoto"}), 400

    # Se non esiste ancora la conversazione: creiamone una
    if conv_id is None:
        title = user_message[:60] + ("..." if len(user_message) > 60 else "")
        now = datetime.utcnow().isoformat(timespec="seconds")
        con = get_db_connection()
        cur = con.cursor()
        cur.execute(
            "INSERT INTO conversations (title, created_at) VALUES (?, ?)",
            (title, now),
        )
        conv_id = cur.lastrowid
        con.commit()
        con.close()

    # Costruzione messaggi per OpenAI
    full_messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    rag_context = ""
    if use_rag:
        rag_context = get_rag_context(user_message)
        if rag_context:
            full_messages.append(
                {
                    "role": "system",
                    "content": (
                        "Queste sono informazioni rilevanti da FAQ e manuale tecnico:\n"
                        f"{rag_context}\n"
                        "Usale come contesto principale per rispondere. "
                        "Se non bastano, dillo apertamente."
                    ),
                }
            )

    # Aggiungiamo la history inviata dal frontend
    full_messages.extend(messages)
    # (il messaggio utente corrente è già incluso in messages lato frontend)

    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=full_messages,
            temperature=0.7,
            max_tokens=400,
        )
        reply = res.choices[0].message.content
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Salvataggio su DB (messaggio utente + risposta)
    now = datetime.utcnow().isoformat(timespec="seconds")
    con = get_db_connection()
    cur = con.cursor()
    cur.execute(
        """
        INSERT INTO messages (conversation_id, role, content, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (conv_id, "user", user_message, now),
    )
    cur.execute(
        """
        INSERT INTO messages (conversation_id, role, content, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (conv_id, "assistant", reply, now),
    )
    con.commit()
    con.close()

    return jsonify(
        {
            "reply": reply,
            "conversation_id": conv_id,
            "used_rag": use_rag,
            "rag_context_present": bool(rag_context),
        }
    )


if __name__ == "__main__":
    init_db()
    ensure_rag_index()  # costruisce o carica l'indice RAG se possibile
    app.run(host="0.0.0.0", port=5000, debug=True)
