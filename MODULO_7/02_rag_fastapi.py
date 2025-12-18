## uvicorn 02_rag_fastapi:app --reload --port 8000
## rag_fastapi.py
'''
ML RAG (Retrieval-Augmented Generation) è una tecnica fondamentale
nell'apprendimento automatico (ML) che potenzia i modelli linguistici di grandi dimensioni (LLM)
collegandoli a basi di conoscenza esterne e aggiornate, permettendo loro di generare risposte
più accurate, pertinenti e "radicate" nei fatti, senza dover essere riaddestrati,
combinando il recupero di informazioni rilevanti con la capacità di generazione del testo. 

Come funziona
Ricerca (Retrieval): Quando un utente pone una domanda, il sistema RAG cerca
in un archivio di documenti esterni (come documenti aziendali, database o il web)
i frammenti di informazioni più rilevanti per la domanda.

Aumento (Augmentation): La domanda originale dell'utente viene "aumentata"
con i frammenti di informazioni recuperati, fornendo contesto aggiuntivo.
Generazione (Generation): L'LLM utilizza sia la domanda originale che i dati
recuperati per generare una risposta più completa, accurata e basata su fonti specifiche. 

Vantaggi principali
Precisione e aggiornamento: Fornisce risposte basate su dati recenti e specifici,
superando i limiti delle conoscenze statiche del modello.

Riduzione delle "allucinazioni": Diminuisce la tendenza degli LLM a inventare informazioni,
poiché sono "ancorati" a fonti reali.

Trasparenza: Permette di citare le fonti utilizzate, aumentando la fiducia nell'output.

Efficienza: Evita costosi riaddestramenti (fine-tuning) del modello,
aggiornando la knowledge base esterna.

Personalizzazione: Permette di creare applicazioni di AI per
domini specifici (es. knowledge base aziendale). 

Applicazioni
Chatbot aziendali e assistenti virtuali che rispondono su dati interni.
Sistemi di Q&A (domande e risposte) che citano le fonti.
Generazione di contenuti basati su documenti specializzati.
'''

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sentence_transformers import SentenceTransformer, util

class Query(BaseModel):
    domanda: str

app = FastAPI()

# Carico base di conoscenza
faq = pd.read_csv("faq_prodotti.csv")
model = SentenceTransformer("hf/all-MiniLM-L6-v2")
faq_emb = model.encode(faq["domanda"].tolist(), convert_to_tensor=True)

@app.post("/ask")
def ask(data: Query):
    query_emb = model.encode(data.domanda, convert_to_tensor=True)
    scores = util.cos_sim(query_emb, faq_emb)[0]

    idx = int(scores.argmax())
    return {
        "domanda_corrispondente": faq.loc[idx, "domanda"],
        "risposta": faq.loc[idx, "risposta"],
        "similarita": float(scores[idx])
    }
