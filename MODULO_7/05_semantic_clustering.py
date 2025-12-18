'''
Il metodo di clustering k-means è una tecnica di apprendimento
automatico non supervisionato utilizzata per identificare cluster di oggetti
dati in un set di dati. Esistono molti metodi diversi,

Clustering partizionale
Clustering gerarchico
Clustering basato sulla densità

Ottimo per raggruppare domande simili.
'''

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

df = pd.read_csv("faq_prodotti.csv")  # colonna "domanda"

model = SentenceTransformer("hf/all-MiniLM-L6-v2")
emb = model.encode(df["domanda"].tolist())



# Esempio semplice
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(emb)

df["cluster"] = labels
print(df)
