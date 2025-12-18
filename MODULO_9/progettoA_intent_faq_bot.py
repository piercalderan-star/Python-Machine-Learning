"""
Progetto A – Intent + FAQ Bot (Transformers + RAG semplice)

- Carica un modello Transformer (es. DistilBERT) fine-tuned o di base
- Carica un piccolo dataset di intent (intents_small.csv)
- Costruisce un mini-RAG su FAQ di esempio
- Fornisce una semplice interfaccia da terminale (CLI) per fare domande

Prerequisiti:
- pip install transformers sentence-transformers torch scikit-learn pandas
- modello HF disponibile localmente (es. ./hf/distilbert-base-uncased)
"""

import os
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder

INTENTS_CSV = "intents_small.csv"
FAQ_CSV = "faq_prodotti.csv"  # opzionale, se disponibile
MODEL_NAME = "sentiment_transformer_model"  # se hai fine-tuning, altrimenti base HF
EMB_MODEL = "hf/all-MiniLM-L6-v2"

def load_intent_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    except Exception:
        print("Impossibile caricare modello fine-tuned, uso distilbert-base-uncased di default.")
        base_name = "distilbert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(base_name)
        model = AutoModelForSequenceClassification.from_pretrained(base_name, num_labels=2)
    return tokenizer, model

def load_intents():
    df = pd.read_csv(INTENTS_CSV)
    le = LabelEncoder()
    df["intent_id"] = le.fit_transform(df["intent"])
    return df, le

def softmax(logits):
    e = np.exp(logits - np.max(logits))
    return e / e.sum()

def predict_intent(text, tokenizer, model, label_encoder):
    import torch
    model.eval()
    enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        out = model(**enc)
        logits = out.logits[0].numpy()
    probs = softmax(logits)
    idx = int(np.argmax(probs))
    intent = label_encoder.inverse_transform([idx])[0]
    return intent, float(probs[idx])

def load_faq_corpus():
    if not os.path.exists(FAQ_CSV):
        print("Attenzione: faq_prodotti.csv non trovato, uso un piccolo corpus di default.")
        faq = [
            "Per contattare l'assistenza puoi scrivere a support@example.com",
            "Il corso di machine learning include 9 moduli principali",
            "Per iscriverti al corso visita la pagina iscrizioni del sito"
        ]
    else:
        df = pd.read_csv(FAQ_CSV, encoding="utf-8", sep=",")
        col = df.columns[0]
        faq = df[col].dropna().tolist()
    return faq

def build_rag_index(docs, model_path=EMB_MODEL):
    model = SentenceTransformer(model_path)
    emb = model.encode(docs)
    return model, docs, emb

def rag_search(query, model, docs, emb, top_k=2):
    q = model.encode([query])[0]
    scores = []
    for e in emb:
        s = float(np.dot(q, e) / ((np.linalg.norm(q)+1e-9)*(np.linalg.norm(e)+1e-9)))
        scores.append(s)
    idx = np.argsort(scores)[::-1][:top_k]
    return [(docs[i], scores[i]) for i in idx]

def main():
    print("=== Progetto A – Intent + FAQ Bot ===")
    intents_df, le = load_intents()
    tokenizer, cls_model = load_intent_model()
    emb_model, faq_docs, faq_emb = build_rag_index(load_faq_corpus())

    print("Digita 'esci' per terminare.\n")
    while True:
        text = input("Tu: ").strip()
        if not text:
            continue
        if text.lower() in ("esci", "quit", "exit"):
            break

        intent, prob = predict_intent(text, tokenizer, cls_model, le)
        print(f"[Intent previsto: {intent} ({prob:.2f})]")

        if intent in ("supporto", "info_corso", "prezzo", "iscrizione"):
            risultati = rag_search(text, emb_model, faq_docs, faq_emb)
            print("FAQ correlate:")
            for doc, score in risultati:
                print(f"- {doc}  (score={score:.3f})")
        else:
            print("Risposta genérica: posso aiutarti con il corso o con i prodotti.")

        print()

if __name__ == "__main__":
    main()
