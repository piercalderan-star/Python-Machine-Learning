'''
Decision Tree (Albero Decisionale) e Random Forest (Foresta Casuale)

Sono due algoritmi fondamentali di machine learning supervisionato,
utilizzati sia per problemi di classificazione che di regressione.
La Random Forest è, essenzialmente, un insieme di molti Alberi Decisionali.

Ecco una spiegazione dettagliata di entrambi.

1. Decision Tree (Albero Decisionale)
Un Albero Decisionale è un modello predittivo
che opera suddividendo ricorsivamente i dati
in sottoinsiemi più piccoli in base a una serie
di domande logiche (soglie sulle caratteristiche),
fino a raggiungere una decisione finale in un "nodo foglia".

Puoi immaginarlo come un diagramma di flusso (flowchart): 
Nodi Radice/Interni:
    Rappresentano le decisioni o i test sulle caratteristiche dei dati
    (es. "Il reddito del cliente è superiore a 50k€?").
    
Rami:
    Rappresentano il risultato di una decisione (Sì/No).
    
Nodi Foglia:
    Rappresentano il risultato finale, la classe predetta o il valore (es. "Acquisterà il prodotto", "Reddito alto"). 
Vantaggi principali:
Facile interpretazione: Sono molto intuitivi e facili da spiegare, specialmente se poco profondi.
Poca preparazione dati: Non richiedono la normalizzazione o la scalatura delle feature. 
Svantaggi principali:
Overfitting: Un singolo albero può diventare troppo complesso e "imparare a memoria" i dati di addestramento, perdendo la capacità di generalizzare bene su nuovi dati (problema di overfitting).
Instabilità: Piccole variazioni nei dati di input possono portare a una struttura dell'albero completamente diversa. 
2. Random Forest (Foresta Casuale)
La Random Forest risolve i problemi di overfitting e instabilità del singolo albero decisionale. Non si basa su un singolo albero, ma costruisce una "foresta" composta da numerosi alberi decisionali individuali leggermente diversi tra loro. 
L'algoritmo utilizza una tecnica chiamata "bagging" (dall'inglese Bootstrap Aggregating) e un campionamento casuale delle feature per creare diversità tra gli alberi: 
Campionamento casuale dei dati (Bagging): Ogni albero viene addestrato su un sottoinsieme di dati scelto casualmente dal set di dati originale (con sostituzione).
Campionamento casuale delle feature: Ad ogni nodo dell'albero, solo un sottoinsieme casuale delle caratteristiche (feature) viene considerato per la migliore suddivisione, non tutte le feature disponibili. 
Come avviene la predizione:
Quando si deve fare una previsione, ogni albero nella foresta esprime il proprio "voto" (per la classificazione) o la propria previsione (per la regressione). Il risultato finale della Random Forest è la media delle previsioni (per regressione) o il voto di maggioranza (per classificazione) di tutti gli alberi. 
Vantaggi principali:
Alta accuratezza: Generalmente offre prestazioni migliori di un singolo albero decisionale.
Robustezza all'overfitting: La combinazione di molti alberi riduce significativamente il rischio di overfitting.
Gestione di molti dati: Funziona bene con set di dati di grandi dimensioni e con molte feature. 
Svantaggi principali:
Complessità e interpretabilità: È molto meno intuitiva da visualizzare e spiegare rispetto a un singolo albero decisionale.
Costo computazionale: Richiede più risorse (memoria e CPU) perché deve addestrare molti alberi. 
In sintesi, la Random Forest è un approccio più potente e robusto, che sacrifica un po' di interpretabilità per ottenere una maggiore precisione predittiva, sfruttando la saggezza della folla (degli alberi).
'''

#Esempio 1 (semplice)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

df = pd.read_csv("titanic_ml.csv")

X = df[["età", "classe", "tariffa"]]
y = df["sopravvissuto"]

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))


'''
Per usare un database in stile Titanic,
che include dati sia numerici (età, tariffa)
che categorici (sesso, classe),
è necessaria una fase di pre-elaborazione dei dati
(data preprocessing) per convertire tutto in formato numerico
prima di applicare gli algoritmi.

L'esempio seguente mostra come:
Creare un DataFrame simulato con dati casuali aggiuntivi.
Pre-elaborare i dati (gestione dei valori mancanti e conversione di "sesso").

Addestrare un modello di Random Forest.
Visualizzare il primo Albero Decisionale della foresta usando matplotlib.
Esempio Python: Random Forest e Titanic Dataset
'''

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)
print("Accuracy RF:", model.score(X_test, y_test))


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score

# 1. Creazione di un database clienti simulato (DataFrame pandas)
np.random.seed(42) # Per riproducibilità

# Dati di esempio iniziali
data = {
    'Età': [22, 38, 26, 35, 35, 54, 2, 27, 14, 4, 45, 21, 48, 63, 23],
    'Sesso': ['uomo', 'donna', 'donna', 'donna', 'uomo', 'uomo', 'uomo', 'uomo', 'donna', 'donna', 'uomo', 'uomo', 'donna', 'uomo', 'uomo'],
    'Classe': [3, 1, 3, 1, 3, 1, 3, 3, 2, 3, 1, 3, 1, 2, 3],
    'Tariffa': [7.25, 71.28, 7.92, 53.1, 8.05, 51.86, 21.07, 11.13, 30.07, 16.7, 26.55, 7.22, 76.73, 13.5, 7.85],
    'Sopravvissuto': [0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0]
}
df = pd.DataFrame(data)

# Aggiunge alcuni dati casuali con valori mancanti per dimostrare il preprocessing
for _ in range(20):
    nuova_riga = {
        'Età': np.random.randint(1, 80) if np.random.rand() > 0.1 else np.nan,
        'Sesso': np.random.choice(['uomo', 'donna']),
        'Classe': np.random.choice([1, 2, 3]),
        'Tariffa': np.random.uniform(5, 150),
        'Sopravvissuto': np.random.choice([0, 1])
    }
    df = pd.concat([df, pd.DataFrame([nuova_riga])], ignore_index=True)

print("Prime 5 righe del Database Originale:")
print(df.head())
print(f"\nValori mancanti prima del preprocessing:\n{df.isnull().sum()}")

# 2. Pre-elaborazione dei dati
# Sostituisce i valori mancanti nell'età con la mediana (un approccio comune)
df['Età'].fillna(df['Età'].median(), inplace=True)

# Converte la colonna 'Sesso' da stringhe a numeri (uomo=0, donna=1) per evitare l'errore di visualizzazione
df['Sesso'] = df['Sesso'].map({'uomo': 0, 'donna': 1}) #

# Definisce le feature (X) e il target (y)
features = ['Età', 'Sesso', 'Classe', 'Tariffa']
X = df[features]
y = df['Sopravvissuto']

# 3. Addestramento del modello Random Forest
# Divide i dati in set di addestramento e test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crea e addestra il classificatore Random Forest (es. con 100 alberi)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Valuta il modello
predictions = rf_model.predict(X_test)
print(f"\nAccuratezza del modello Random Forest: {accuracy_score(y_test, predictions):.2f}")

# 4. Visualizzazione di un singolo Albero Decisionale della foresta
# Una Random Forest ha molti alberi, ne visualizziamo solo il primo (indice 0) per mostrare l'architettura.
fig = plt.figure(figsize=(10, 10))
plot_tree(rf_model.estimators_[0], # Il primo albero della foresta
          feature_names=features,
          class_names=['Deceduto', 'Sopravvissuto'],
          filled=True,
          impurity=True,
          rounded=True,
          fontsize=8)
plt.title("Primo Albero Decisionale nella Random Forest (Titanic Dataset)")
plt.show()

'''
Note sull'output
Il grafico dell'albero:
    Apparirà una finestra con un diagramma di flusso
    con solo il primo albero della foresta.

Complessità:
    Anche un singolo albero può essere molto grande
    e complesso se non si limita la profondità (max_depth).

Interpretazione Nodi:
    Ogni nodo mostra la condizione di split
    (es. Sesso <= 0.5), l'indice di Gini (impurezza),
    il numero di campioni (samples),
    e la distribuzione delle classi (value).
'''


'''
L'indice di Gini (o Gini Impurity, impurità di Gini)
è un concetto fondamentale utilizzato specificamente
negli alberi decisionali
(e di conseguenza nelle Random Forest)
per determinare quanto "puro" o "mescolato" sia un insieme di dati
in un nodo specifico. 
In termini semplici,
l'indice di Gini misura la probabilità che un
elemento scelto casualmente da un set di dati
venga etichettato in modo errato se fosse etichettato
in modo casuale in base alla distribuzione delle etichette nel sottoinsieme. 

A cosa serve? 
Quando un algoritmo come l'albero decisionale deve scegliere
come suddividere i dati in un nodo
(ad esempio, "dividere per età > 30 anni" o "dividere per sesso"),
valuta l'indice di Gini di tutte le possibili divisioni. 
L'obiettivo dell'algoritmo è scegliere la divisione che produce il maggior decremento
dell'impurità di Gini, ovvero la divisione che rende i nodi figli il più "puri" possibile. 

Come funziona la metrica
L'indice di Gini è un valore compreso tra 0 e 0.5 (o 1.0, a seconda della normalizzazione): 
Gini = 0 (Puro):
    Significa che tutti gli elementi nel nodo appartengono alla stessa singola classe.
    Questo è l'ideale: il nodo è "completo" e non necessita di ulteriori divisioni (è un nodo foglia).

Gini = 0.5 (Massima Impurità):
    Significa che le classi sono distribuite in modo perfettamente
    uniforme all'interno del nodo (ad esempio, in un problema binario, 50% di una classe e 50% dell'altra). 

Esempio Rapido (Titanic) 
Consideriamo un nodo nel nostro esempio Titanic: 
Nodo A (Puro) Gini=0:
    Contiene 20 passeggeri, e tutti e 20 sono Sopravvissuti. Ottimo.
Nodo B (Impuro) Gini=0.5:
    Contiene 20 passeggeri, di cui 10 Sopravvissuti e 10 Deceduti. Massima impurità.
Nodo C (Intermedio) Gini 0.32:
    Contiene 20 passeggeri, di cui 16 Sopravvissuti e 4 Deceduti. Moderatamente impuro. 

L'algoritmo cercherà sempre di passare da nodi con Gini alto a nodi con Gini basso. 
'''

'''
Gini si riferisce a una persona:
    Corrado Gini, uno statistico, sociologo ed economista italiano.
    L'indice di Gini è stato sviluppato originariamente da lui nel 1912.
    Tuttavia, nella sua forma originale, era utilizzato in economia per misurare
    la disuguaglianza nella distribuzione del reddito o della ricchezza
    all'interno di una nazione (World Population Review on Gini Coefficient,
    Fondazione Corrado Gini Website)
    https://worldpopulationreview.com/country-rankings/gini-coefficient-by-country
    
Nel machine learning, il concetto di disuguaglianza o "mescolanza"
delle classi è stato adattato per misurare l'impurità (o "disuguaglianza tra le classi")
all'interno dei nodi degli alberi decisionali, come spiegato in precedenza.
'''




















