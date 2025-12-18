'''
LM zero-shot si riferisce alla capacità di un modello linguistico
(LM, Language Model) di eseguire un compito per il quale
non è stato esplicitamente addestrato,
utilizzando solo la conoscenza acquisita durante il pre-addestramento. 

Ecco i punti chiave:
Modello Linguistico (LM):
 Un modello di intelligenza artificiale addestrato su grandi quantità di testo
 per comprendere e generare linguaggio umano.

Zero-Shot (Zero Colpi/Esempi):
 Il modello non riceve alcun esempio specifico del compito da svolgere
 al momento dell'inferenza.
 Gli viene fornita solo un'istruzione testuale (prompt).

Ad esempio, si può chiedere a un modello di tradurre una frase,
riassumere un testo o classificare un sentimento semplicemente
descrivendo il compito nel prompt,
senza avergli mai mostrato esempi specifici
di "traduzione zero-shot" durante l'addestramento.
Il modello sfrutta la sua vasta conoscenza generale
per dedurre come completare l'attività.
'''

##Esempio 1
##semplice stampa degli score (risultati)
##from transformers import pipeline
##
##classifier = pipeline("zero-shot-classification",
##                      model="facebook/bart-large-mnli")
##
##text = "Il telefono non si accende più e non risponde ai comandi."
##
##labels = ["hardware", "software", "batteria", "rete"]
##
##res = classifier(text, candidate_labels=labels)
##print(res)
##

##Esempio 2
##from transformers import pipeline
##
### 1. Carica la pipeline di classificazione zero-shot specificando il modello.
###    Il modello "facebook/bart-large-mnli" è ottimizzato per questo compito.
##classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
##
### 2. Definisci il testo da classificare (la "premessa")
##testo_da_classificare = "Il profitto trimestrale dell'azienda è aumentato del 20%, superando le aspettative del mercato."
##
### 3. Definisci le etichette candidate (le "ipotesi")
###    Puoi usare qualsiasi etichetta descrittiva.
##etichette_candidate = ["finanza", "sport", "politica", "tecnologia", "intrattenimento"]
##
### 4. Esegui la classificazione
###    Il modello valuta la relazione di "implicazione" (entailment) tra il testo e ogni etichetta.
##risultato = classifier(testo_da_classificare, etichette_candidate)
##
### 5. Stampa i risultati
##print("Testo originale:", risultato['sequence'])
##print("Etichette e probabilità:")
##for label, score in zip(risultato['labels'], risultato['scores']):
##    print(f"- {label}: {score:.4f}")
##
### Puoi anche stampare l'intero dizionario dei risultati per vedere l'ordinamento
### print("\nRisultato completo:", risultato)

##Esempio 3 con grafico
import matplotlib.pyplot as plt
from transformers import pipeline

# --- Parte 1: Classificazione Zero-Shot (come prima) ---
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

testo_da_classificare = "Il profitto trimestrale dell'azienda è aumentato del 20%, superando le aspettative del mercato."
etichette_candidate = ["finanza", "sport", "politica", "tecnologia", "intrattenimento"]

risultato = classifier(testo_da_classificare, etichette_candidate)

# Estrai etichette e score dai risultati per il grafico
labels = risultato['labels']
scores = risultato['scores']

# --- Parte 2: Visualizzazione con Matplotlib ---

# Ordina i dati dal punteggio più basso al più alto per un grafico a barre orizzontali pulito
# (I risultati della pipeline sono già ordinati dal più alto al più basso, li invertiamo per il grafico a barre)
labels.reverse()
scores.reverse()

# Crea il grafico
plt.figure(figsize=(10, 6))
plt.barh(labels, scores, color='skyblue')
plt.xlabel('Punteggio di Probabilità')
plt.title('Classificazione Zero-Shot del Testo')
plt.xlim(0, 1) # Imposta l'asse X da 0 a 1 (probabilità)

# Aggiunge i valori esatti accanto alle barre
for index, value in enumerate(scores):
    plt.text(value, index, f'{value:.4f}', va='center')

# Mostra il grafico
plt.tight_layout()
plt.show()


####Esempio 4 con grafico e csv
##import matplotlib.pyplot as plt
##import pandas as pd
##from transformers import pipeline
##import os
##
### --- Parte 1: Classificazione Zero-Shot ---
##classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
##
##testo_da_classificare = "Il profitto trimestrale dell'azienda è aumentato del 20%, superando le aspettative del mercato."
##etichette_candidate = ["finanza", "sport", "politica", "tecnologia", "intrattenimento"]
##
##risultato = classifier(testo_da_classificare, etichette_candidate)
##
##labels = risultato['labels']
##scores = risultato['scores']
##
### --- Parte 2: Salvataggio Automatico in CSV con Pandas ---
##
### Crea un DataFrame Pandas usando i risultati
##df_risultati = pd.DataFrame({
##    'Etichetta': labels,
##    'Probabilita': scores
##})
##
### Definisci il nome del file CSV
##nome_file_csv = 'risultati_classificazione.csv'
##
### Salva il DataFrame in un file CSV nella root dello script
### Usiamo index=False per evitare di scrivere l'indice di riga di Pandas nel file
##df_risultati.to_csv(nome_file_csv, index=False)
##
##print(f"Risultati salvati con successo nel file: {os.path.abspath(nome_file_csv)}")
##
### --- Parte 3: Visualizzazione con Matplotlib (Grafico a barre) ---
##
### Ordina i dati dal punteggio più basso al più alto per un grafico a barre orizzontali pulito
### (i risultati della pipeline sono già ordinati dal più alto al più basso, li invertiamo per il grafico)
##labels.reverse()
##scores.reverse()
##
##plt.figure(figsize=(10, 6))
##plt.barh(labels, scores, color='skyblue')
##plt.xlabel('Punteggio di Probabilità')
##plt.title('Classificazione Zero-Shot del Testo')
##plt.xlim(0, 1) # Imposta l'asse X da 0 a 1 (probabilità)
##
### Aggiunge i valori esatti accanto alle barre
##for index, value in enumerate(scores):
##    plt.text(value, index, f'{value:.4f}', va='center')
##
##plt.tight_layout()
##plt.show()
##



##Esempio 5 grafico + colori + csv
##import matplotlib.pyplot as plt
##import pandas as pd
##from transformers import pipeline
##import os
##import matplotlib.colors as mcolors
##import numpy as np
##
### --- Parte 1: Classificazione Zero-Shot ---
##classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
##testo_da_classificare = "Il profitto trimestrale dell'azienda è aumentato del 20%, superando le aspettative del mercato."
##etichette_candidate = ["finanza", "sport", "politica", "tecnologia", "intrattenimento"]
##risultato = classifier(testo_da_classificare, etichette_candidate)
##
##labels = risultato['labels']
##scores = risultato['scores']
##
### --- Parte 2: Salvataggio Automatico in CSV con Pandas ---
##df_risultati = pd.DataFrame({'Etichetta': labels, 'Probabilita': scores})
##nome_file_csv = 'risultati_classificazione.csv'
##df_risultati.to_csv(nome_file_csv, index=False)
##print(f"Risultati salvati con successo nel file: {os.path.abspath(nome_file_csv)}")
##
### --- Parte 3: Visualizzazione con Matplotlib e Colori Dinamici ---
##
### Ordina i dati dal punteggio più basso al più alto per il grafico orizzontale
### (I risultati della pipeline sono già ordinati dal più alto al più basso)
##labels.reverse()
##scores.reverse()
##
### Genera una colormap: da Rosso (alto) a Verde chiaro (basso)
### Usiamo 'RdYlGn' e la invertiamo (_r) per avere il rosso in alto/destra (punteggio alto)
##cmap = plt.get_cmap('RdYlGn')
### Normalizza i punteggi da 0 a 1 per mappatura del colore
##norm = mcolors.Normalize(vmin=min(scores), vmax=max(scores))
##colors = [cmap(norm(score)) for score in scores]
##
##plt.figure(figsize=(10, 6))
##
### Usa la lista di colori generata per colorare le barre
##plt.barh(labels, scores, color=colors)
##
##plt.xlabel('Punteggio di Probabilità')
##plt.title('Classificazione Zero-Shot del Testo con Colori Dinamici')
##plt.xlim(0, 1)
##
### Aggiunge i valori esatti accanto alle barre
##for index, value in enumerate(scores):
##    plt.text(value + 0.02, index, f'{value:.4f}', va='center') # Aggiunto offset per migliore visibilità
##
##plt.tight_layout()
##plt.show()
