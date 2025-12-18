'''
Nel machine learning, un Sentence Transformer (trasformatore di frasi)
è un modello specializzato di rete neurale progettato per convertire
intere frasi, paragrafi o immagini in dense rappresentazioni numeriche
(chiamate embedding vettoriali) che ne preservano il significato semantico. 

Funzionamento Principale
A differenza dei modelli linguistici tradizionali
(come le implementazioni originali di BERT) che si
concentrano sugli embedding a livello di parola e
richiedono un'ulteriore elaborazione
(come il mean pooling) per produrre un singolo vettore di frase,
i sentence transformer sono specificamente ottimizzati per questo compito.

Il framework si basa tipicamente su architetture
a rete siamese o a triplette,

(una rete siamese "siamese neural network"
è un tipo specializzato di rete neurale artificiale
che utilizza due reti "gemelle" identiche per confrontare
due input distinti e determinare la loro somiglianza
o dissomiglianza

addestrate per mappare semanticamente testi simili vicini tra loro
in uno spazio vettoriale,
e testi dissimili lontani. Questa disposizione rende il calcolo
della similarità,
spesso tramite similarità del coseno, molto efficiente e veloce. 

Applicazioni Comuni
La libreria sentence-transformers in Python,
ora parte dell'ecosistema Hugging Face,
offre un modo semplice per accedere e utilizzare una vasta gamma di modelli pre-addestrati.

Le applicazioni includono: 
Ricerca Semantica:
    Trovare documenti o risposte a domande basati sul significato,
    superando la semplice corrispondenza di parole chiave.
    
Similarità Testuale:
    Determinare quanto due frasi o documenti siano simili nel loro
    intento o contenuto (ad esempio, per rilevare domande duplicate).
    
Clustering:
    Raggruppare automaticamente documenti o frasi con
    argomenti simili senza etichette predefinite.

Estrazione di Parafrasi:
    Identificare frasi diverse che esprimono lo stesso significato.

Classificazione:
    Utilizzare gli embedding generati come funzionalità
    di input per modelli di classificazione
    (ad esempio, per l'analisi del sentiment).
'''

from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("hf/all-MiniLM-L6-v2")

sentences = [
    "Il gatto dorme sul divano.",
    "Un felino sta riposando.",
    "Oggi piove molto forte."
]

emb = model.encode(sentences)
print("Shape:", emb.shape)
print("Embedding esempio (primi 5 valori):", emb[0][:5])






