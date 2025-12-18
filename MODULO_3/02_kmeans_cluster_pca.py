'''
PCA
La visualizzazione PCA (Analisi delle Componenti Principali)
è una tecnica per ridurre la complessità dei dati
ad alta dimensionalità in grafici a dispersione 2D o 3D,
usando le prime due o tre componenti principali (PC1, PC2, PC3)
come nuovi assi per mostrare pattern, cluster e anomalie,
semplificando set di dati complessi e rendendoli più
interpretabili e visualizzabili.

Come funziona la visualizzazione PCA
Trasformazione: La PCA trasforma le variabili originali correlate
in un nuovo set di variabili non correlate (le componenti principali).

Riduzione della dimensionalità:
    Si selezionano le prime componenti (PC1, PC2, ecc.)
    che catturano la maggior parte della varianza dei dati, riducendo il numero di dimensioni.

Grafico a dispersione:
    Si crea un grafico a dispersione dove l'asse X è la PC1 e l'asse Y è la PC2,
    proiettando i dati originali in questo spazio ridotto. 

A cosa serve la visualizzazione PCA
Esplorazione dati:
    Aiuta a capire la struttura interna dei dati,
    identificando raggruppamenti o pattern nascosti.

Identificazione anomalie:
    I punti dati che si allontanano dal gruppo principale
    sono più facili da individuare.
    
Visualizzazione:
    Permette di vedere dati complessi in 2D,
    rendendoli più comprensibili di quanto lo sarebbero in molte dimensioni.
    
Pre-elaborazione:
    Riduce il rumore e la complessità prima di utilizzare altri algoritmi di machine learning. 

Esempi di applicazione
Settore sanitario:
    Per la diagnosi precoce di malattie.
Immagini:
    Per il riconoscimento facciale, riducendo le dimensioni delle immagini.
Ricerca di mercato:
    Per identificare tendenze e comportamenti dei consumatori.
Bioinformatica:
    Per analizzare grandi set di dati biologici.
'''

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df = pd.read_csv("clienti_cluster.csv")
X = df[["spesa_annua", "frequenza_visite", "punteggio_fedelta"]]

# Addestramento K-Means
model = KMeans(n_clusters=2)
model.fit(X)

df["cluster"] = model.labels_

plt.scatter(df["spesa_annua"], df["frequenza_visite"], c=df["cluster"])
plt.xlabel("Spesa annua")
plt.ylabel("Frequenza visite")
plt.show()


#pca_visualization.py
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.scatter(X_pca[:,0], X_pca[:,1], c=df["cluster"])
plt.title("PCA Visualization")
plt.show()










