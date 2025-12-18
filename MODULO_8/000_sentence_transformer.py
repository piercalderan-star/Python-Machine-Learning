from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt

# 1. Caricamento del modello pre-addestrato
# 'paraphrase-multilingual-MiniLM-L12-v2' è un modello efficace che supporta l'italiano.
print("Caricamento del modello Sentence Transformer...")
model = SentenceTransformer('hf/paraphrase-multilingual-MiniLM-L12-v2')
print("Modello caricato.")

# 2. Il nostro "database" di frasi (Corpus)
corpus = [
    "Il gatto nero sta dormendo sul divano.",
    "Un cane marrone gioca nel parco.",
    "La macchina rossa corre veloce sull'autostrada.",
    "Ho comprato una nuova auto sportiva.",
    "Il mio animale domestico preferito è un felino.",
    "Gli animali dormono molto durante l'inverno.",
    "La CPU del computer è molto calda.",
    "Ieri sono andato a fare la spesa al supermercato."
]

# 3. Creazione degli embeddings (vettori numerici) per il corpus
print(f"Creazione embeddings per {len(corpus)} frasi...")
corpus_embeddings = model.encode(corpus)

print("Shape degli embeddings:", corpus_embeddings.shape) # Sarà N_frasi x Dimensione_vettore (es. 8 x 384)

# --- 4. Input Utente e Ricerca della similarità ---

def find_similar_sentence(user_input):
    # Crea l'embedding per la frase di input dell'utente
    input_embedding = model.encode([user_input])
    
    # Calcola la similarità del coseno tra l'input e tutte le frasi del corpus
    # La similarità del coseno varia tra -1 (opposto) e 1 (identico)
    similarities = cosine_similarity(input_embedding, corpus_embeddings)
    
    # Trova l'indice della frase con la similarità maggiore
    best_match_index = np.argmax(similarities)
    best_score = similarities[0, best_match_index]
    best_sentence = corpus[best_match_index]
    
    return best_sentence, best_score, similarities[0]

# --- Esecuzione dell'esempio ---

user_query = "Qualcosa sull'animale domestico che riposa"
best_sentence, best_score, all_scores = find_similar_sentence(user_query)

print(f"\n--- Analisi Query ---")
print(f"Query Utente: '{user_query}'")
print(f"Frase più simile trovata: '{best_sentence}'")
print(f"Punteggio di similarità (coseno): {best_score:.4f}")

# Visualizzazione opzionale dei punteggi (con matplotlib)
plt.figure(figsize=(10, 5))
plt.bar(range(len(corpus)), all_scores)
plt.xticks(range(len(corpus)), corpus, rotation=90)
plt.ylabel('Punteggio Similarità Coseno')
plt.title('Similarità tra Query e Frasi del Corpus')
plt.tight_layout()
plt.show()

# Esempio 2:
user_query_2 = "Sono andato a comprare cibo"
best_sentence_2, _, _ = find_similar_sentence(user_query_2)
print(f"\nQuery Utente 2: '{user_query_2}'")
print(f"Frase più simile trovata: '{best_sentence_2}'")
