'''
Il Gradient Boosting è un'altra tecnica potente nel machine learning che,
come la Random Forest, combina più alberi decisionali deboli per creare
un modello forte.
A differenza del bagging (usato in Random Forest),
che addestra gli alberi in parallelo,
il Gradient Boosting li addestra sequenzialmente.
Ogni nuovo albero cerca di correggere gli errori (i residui)
commessi dagli alberi precedenti.
L'algoritmo più popolare e performante per il Gradient Boosting
è l'XGBoost (eXtreme Gradient Boosting), che useremo in questo esempio.
'''

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Assicurati che il file CSV esista (eseguendo l'esempio Random Forest precedente se necessario)
try:
    df = pd.read_csv('prezzi_case_ml.csv')
except FileNotFoundError:
    print("Il file 'prezzi_case_ml.csv' non è stato trovato. Si prega di crearlo.")
    exit()

print("\nPrime 5 righe del dataset:")
print(df.head())

# Definisci le feature (X) e il target (y)
features = ['superficie', 'stanze', 'zona']
X = df[features]
y = df['prezzo']

# Suddivisione in Training e Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Addestramento del modello XGBoost Regressor
# L'XGBoost ha molti parametri, qui usiamo un set base
model_xgb = xgb.XGBRegressor(objective='reg:squarederror', # Funzione obiettivo per la regressione
                             n_estimators=100,           # Numero di alberi
                             learning_rate=0.1,          # Tasso di apprendimento (quanto corregge ogni albero)
                             random_state=42)

model_xgb.fit(X_train, y_train)

print("\nAddestramento del modello XGBoost completato.")

# 4. Valutazione
predictions = model_xgb.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Errore Assoluto Medio (MAE) sul set di test: {mae:.2f} €")
print(f"Coefficiente R^2 (Bontà del modello): {r2:.2f}")

# 5. Visualizzazione delle Feature Importance
# Un grande vantaggio degli algoritmi ad albero è la capacità di mostrare quali feature
# sono state più importanti per le decisioni del modello.

feature_importance = pd.Series(model_xgb.feature_importances_, index=features).sort_values(ascending=False)

plt.figure(figsize=(8, 5))
feature_importance.plot(kind='barh')
plt.title("Importanza delle Feature (XGBoost)")
plt.xlabel("Importanza Relativa")
plt.ylabel("Feature")
plt.show()

# --- 6. Esempio di Previsione Singola ---
# Previsione per una casa di 90 mq, 3 stanze, in centro (Zona 1)
##sup=int(input("superficie: "))
##sta=int(input("stanze: "))
##zon=int(input("zona 1 centro 2 periferia: "))

nuova_casa = pd.DataFrame([[90, 3, 1]], columns=features)
prezzo_previsto = model_xgb.predict(nuova_casa)

print(f"\nPrezzo previsto per una casa 90mq, 3 stanze, Centro: {prezzo_previsto[0]:.2f} €")
