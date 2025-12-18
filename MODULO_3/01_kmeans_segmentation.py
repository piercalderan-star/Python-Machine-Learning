'''
K-means
K-means è un algoritmo di apprendimento automatico
non supervisionato che raggruppa i dati in K cluster distinti,
dove ogni punto dati appartiene al cluster con il centroide (la media)
più vicino, minimizzando la varianza all'interno di ogni gruppo.
È ampiamente utilizzato per scoprire schemi nei dati non etichettati,
suddividendoli in sottogruppi omogenei in base alla similarità,
con applicazioni che vanno dall'analisi di dati GPS
alla segmentazione di immagini. 

Come funziona in sintesi
    Scegliere K:
    Si decide in anticipo il numero di cluster (K) desiderato.

    Inizializzazione:
        Si posizionano K centroidi (punti rappresentativi)
        casualmente tra i dati.
        
Assegnazione:
    Ogni punto dati viene assegnato al centroide più vicino (distanza euclidea).
    
Aggiornamento:
    I centroidi vengono ricalcolati come la media di tutti i punti assegnati
    al loro cluster.

Iterazione:
    I passaggi 3 e 4 vengono ripetuti finché i centroidi non
    si spostano più in modo significativo,
    raggiungendo un'ottimizzazione. 

Caratteristiche principali:
Apprendimento Non Supervisionato:
    Non richiede dati etichettati per l'addestramento.
Basato su Centroidi:
    Ogni cluster è rappresentato dalla sua media (centroide).
Partizione:
    Divide i dati in gruppi che non si sovrappongono.
Obiettivo:
    Minimizzare la somma delle distanze al quadrato
    tra i punti e i centroidi dei loro cluster. 

Utilizzi comuni:
Segmentazione clienti.
Riconoscimento di pattern nei dati.
Compressione o pixelizzazione di immagini.
'''

'''
Esempio in Python:
    Segmentazione Clienti
Questo codice simula un database clienti (utilizzando un DataFrame pandas)
e applica K-means per identificare 3 diversi segmenti di clienti basati
su due caratteristiche:
    "Reddito Annuale"
    "Punteggio di Spesa".
'''

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

# 1. Creazione di un database clienti simulato (DataFrame pandas)
# Solitamente si caricano dati da un file CSV o da un database
data = {
    'ID Cliente': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    'Reddito Annuale (k€)': [15, 20, 25, 30, 32, 35, 40, 45, 50, 60, 70, 80, 85, 90, 95],
    'Punteggio di Spesa (1-100)': [85, 80, 90, 10, 15, 8, 95, 12, 20, 30, 88, 92, 10, 15, 12]
}

df = pd.DataFrame(data)

print("Database Clienti Originale:")
print(df)

# 2. Selezione delle feature per il clustering
# Per questo esempio, usiamo solo due colonne per facilitare la visualizzazione.
X = df[['Reddito Annuale (k€)', 'Punteggio di Spesa (1-100)']].values

# 3. Applicazione dell'algoritmo K-means
# Definiamo il numero di cluster (k) desiderato, ad es. k=3
k = 3
kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10) # n_init raccomandato

# Addestramento del modello sui dati
kmeans.fit(X)

# Ottenimento delle etichette (cluster) assegnate a ciascun cliente
labels = kmeans.labels_

# Aggiunta delle etichette al DataFrame originale per l'analisi
df['Cluster'] = labels

print("\nDatabase Clienti con Cluster Assegnati:")
print(df)

# 4. Visualizzazione dei risultati
plt.figure(figsize=(8, 6))
# I colori dipendono dall'etichetta del cluster assegnata
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k', s=100)

# Disegno dei centroidi (i centri dei cluster)
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, marker='*', c='red', label='Centroidi')

plt.title(f'Segmentazione Clienti con K-Means (k={k})')
plt.xlabel('Reddito Annuale (k€)')
plt.ylabel('Punteggio di Spesa (1-100)')
plt.legend()
plt.grid(True)
plt.show()



'''
Interpretazione dell'output
Il grafico visualizza i clienti come punti colorati
e i centroidi come stelle rosse.

Con k=3 si vedrà probabilmente:

Cluster 0 (es. in basso a sinistra/destra):
    Clienti con reddito basso e punteggio di spesa basso.
        
Cluster 1 (es. in alto a sinistra/destra):
    Clienti con punteggio di spesa alto (indipendentemente
    dal reddito in questo specifico set di dati simulato).

Cluster 2 (es. al centro):
    Clienti con reddito e punteggio di spesa moderati.
    
Questo tipo di segmentazione permette all'azienda
di personalizzare le strategie di marketing per ciascun gruppo.
'''
