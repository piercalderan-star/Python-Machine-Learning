"""
Regressione lineare multipla: prezzo = f(superficie, stanze, zona)

ColumnTransformer
Applica i trasformatori alle colonne di un array o di un DataFrame pandas.
Questo stimatore consente di trasformare separatamente
diverse colonne o sottoinsiemi di colonne dell'input e
le feature generate da ciascun trasformatore
verranno concatenate per formare un unico spazio di feature.
Questo è utile per dati eterogenei o colonnari,
per combinare diversi meccanismi di estrazione di
feature o trasformazioni in un unico trasformatore.

OneHotEncoder
Codifica le caratteristiche categoriali come un array numerico one-hot.
L'input di questo trasformatore dovrebbe essere
un array di interi o stringhe,
che denota i valori assunti dalle caratteristiche categoriche (discrete).
Le caratteristiche sono codificate utilizzando uno schema di codifica one-hot
(noto anche come "uno-di-K" o "fittizio").
Questo crea una colonna binaria per ogni categoria
e restituisce una matrice sparsa o un array denso
(a seconda del sparse_output parametro).
Per impostazione predefinita, l'encoder ricava
le categorie in base ai valori univoci
di ciascuna feature. In alternativa,
è anche possibile specificare categorie manualmente.

"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("prezzi_case_ml.csv")

X = df[["superficie", "stanze", "zona"]]
y = df["prezzo"]

print(X)
print(X[:1][:1])
print(y)

# Encoding zona (categorica)
ct = ColumnTransformer(
    [("zona_enc", OneHotEncoder(), ["zona"])],
    remainder="passthrough"
)

X_enc = ct.fit_transform(X)
print(X_enc)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_enc, y)
print(X_train)
print(y_train)

model = LinearRegression()
model.fit(X_train, y_train)
print("Coefficienti:", model.coef_)
##Cos'è il modello coef_?
##coef_ fornisce un array di pesi stimati
##tramite regressione lineare.
##Ha una forma (n_target, n_feature).
##In questo caso è un array 1D poiché ha un solo target.

#Grafico
plt.plot(y,X_enc)
plt.ylabel("x_enc")
plt.xlabel("prezzo")
plt.title("Regressione lineare multipla")
plt.show()



















