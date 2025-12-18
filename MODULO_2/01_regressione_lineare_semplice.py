"""
Regressione lineare semplice: prezzo = f(superficie)

https://scikit-learn.org/stable/index.html

metodo train_test_split
sklearn.model_selection.train_test_split(*arrays, test_size=None,
train_size=None, random_state=None, shuffle=True, stratify=None)[source]

Suddivide array o matrici in sottoinsiemi casuali di addestramento e test.
Utilità rapida che racchiude la convalida dell'input,
next(ShuffleSplit().split(X, y)) e l'applicazione per l'input
dei dati in un'unica chiamata per suddividere
(e facoltativamente sottocampionare) i dati in un one-liner.

Parametri:
*arrays
sequenza di indicizzabili con la stessa lunghezza/forma[0]
Gli input consentiti sono liste, array numpy, matrici scipy-sparse o dataframe pandas.

test_size
float o int, default=None
Se float, deve essere compreso tra 0,0 e 1,0 e
rappresentare la proporzione del dataset
da includere nella suddivisione del test.
Se int, rappresenta il numero assoluto di campioni di test.
Se None, il valore viene impostato sul complemento della dimensione del train.
Se train_size è anche None, verrà impostato su 0,25.

train_size
float o int, default=None
Se float, deve essere compreso tra 0,0 e 1,0
e rappresentare la proporzione del dataset da includere
nella suddivisione del train.
Se int, rappresenta il numero assoluto di campioni di train.
Se None, il valore viene automaticamente impostato sul complemento della dimensione del test.

random_state
int, istanza di RandomState o None, default=None
Controlla il rimescolamento applicato ai dati prima di applicare la suddivisione.
Passare un int per un output riproducibile tra più chiamate di funzione.

shuffle
bool, default=True
Se mescolare o meno i dati prima della suddivisione.
Se shuffle=False, stratify deve essere None.

stratify
array-like, default=None
Se diverso da None, i dati vengono suddivisi in modo stratificato,
utilizzando questo come etichette di classe.

Restituisce:
splittinglist, length=2 * len(arrays)
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv("prezzi_case_ml.csv")

# Feature e target
X = df[["superficie"]]     # variabile indipendente
y = df["prezzo"]           # variabile dipendente

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)



# Modello
model = LinearRegression()
model.fit(X_train, y_train)

print("Coefficiente =", model.coef_[0])
print("Intercetta array=", model.intercept_) #Termine indipendente nel modello lineare.

##Cos'è il modello coef_?
##coef_ fornisce un array di pesi stimati
##tramite regressione lineare.
##Ha una forma (n_target, n_feature).
##In questo caso è un array 1D poiché ha un solo target.


# Predizione
y_pred = model.predict(X_test)
print("y_test",y_test)
print("y_pred", y_pred)

# Valutazione
mse = mean_squared_error(y_test, y_pred)

'''
Parametri chiave di mean_squared_error
I parametri principali per la funzione sono:
y_true: simile a un array di forma (n_samples,) o (n_samples, n_outputs).
y_pred: simile a un array di forma (n_samples,) o (n_samples, n_outputs).
sample_weight: pesi opzionali per singoli campioni.
multioutput: definisce come aggregare gli errori
per più valori di output (le opzioni includono 'uniform_average', 'raw_values').
Per comodità, scikit-learn fornisce anche
root_mean_squared_error,
che restituisce la radice quadrata dell'MSE,
rimettendo l'errore nelle stesse unità del target
'''

print("MSE:", mse)
print("X_test", X_test)

# Grafico
plt.plot(X_test, y_test, color="blue", label="reale")
plt.plot(X_test, y_pred, color="red", label="predetto")
plt.xlabel("Superficie")
plt.ylabel("Prezzo")
plt.legend()
plt.show()



















