'''
L'algoritmo Random Forest è un'ottima scelta per la regressione
(previsione di un valore numerico come il prezzo di una casa)
perché è robusto e gestisce bene diversi tipi di feature.
A differenza dell'esempio PyTorch, che richiede
la standardizzazione manuale e un ciclo di training esplicito,
Random Forest con scikit-learn è molto più semplice da implementare.
'''

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import os

# 1. Creazione di un database CSV fittizio
def create_house_prices_csv():
    data = {
        'Metri Quadrati': np.random.randint(40, 150, 50),
        'Stanze': np.random.randint(1, 5, 50),
        # Zona: 0 per periferia, 1 per centro
        'Zona': np.random.randint(0, 2, 50),
        'Prezzo':np.random.randint(40000,250000, 50) 
    }
    # Calcola un prezzo base più un fattore casuale e un bonus per la zona centrale
    data['Prezzo'] = (data['Metri Quadrati'] * 1500 + 
                      data['Stanze'] * 5000 + 
                      data['Zona'] * 20000 + 
                      np.random.normal(0, 10000, 50)).astype(int)
                      
    df = pd.DataFrame(data)
    df.to_csv('prezzi_case.csv', index=False)
    print("Creato il file 'prezzi_case.csv'")

create_house_prices_csv()

# 2. Caricamento e Preparazione dei Dati
df = pd.read_csv('prezzi_case.csv')
print("\nPrime 5 righe del dataset:")
print(df.head())

# Definisci le feature (X) e il target (y)
features = ['Metri Quadrati', 'Stanze', 'Zona']
X = df[features]
y = df['Prezzo']

# Suddivisione in Training e Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Addestramento del modello Random Forest Regressor
# Usiamo 100 alberi decisionali per la regressione
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

print("\nAddestramento del modello Random Forest completato.")

# 4. Valutazione (opzionale)
predictions = rf_model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f"Errore Assoluto Medio (MAE) sul set di test: {mae:.2f} €")

# 5. Salvataggio del modello addestrato in un file (opzionale ma utile)
joblib.dump(rf_model, 'modello_prezzi_casa_rf.pkl')
print("Modello salvato come 'modello_prezzi_casa_rf.pkl'")

# --- 6. Input Utente e Previsione Finale ---

print("\n--- Inserisci i dati per la previsione del prezzo ---")

# Carica il modello salvato per l'uso (se lo script viene eseguito in un secondo momento)
loaded_model = joblib.load('modello_prezzi_casa_rf.pkl')

try:
    mq = float(input("Inserisci i metri quadrati: "))
    stanze = float(input("Inserisci il numero di stanze: "))
    # Mappa l'input testuale dell'utente a un valore numerico che il modello capisce (0 o 1)
    zona_input = input("Inserisci la zona (Centro / Periferia): ").strip().lower()
    
    if 'centro' in zona_input:
        zona = 1
    elif 'periferia' in zona_input:
        zona = 0
    else:
        print("Zona non valida inserita. Predefinito a Periferia (0).")
        zona = 0

    # Prepara l'input utente nel formato corretto (DataFrame o array 2D)
    user_input_data = pd.DataFrame({
        'Metri Quadrati': [mq],
        'Stanze': [stanze],
        'Zona': [zona]
    })

    # Esegui la previsione
    predicted_price = loaded_model.predict(user_input_data)
    
    print(f"\nRisultato della previsione:")
    print(f"Il prezzo previsto per la casa è di circa: {predicted_price[0]:.2f} €")

except ValueError:
    print("Errore nell'input. Assicurati di inserire valori numerici validi.")
except Exception as e:
    print(f"Si è verificato un errore: {e}")

