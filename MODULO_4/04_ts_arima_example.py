'''
ARIMA (AutoRegressive Integrated Moving Average) è un modello statistico
potente usato per l'analisi e la previsione delle serie temporali,
che combina tre componenti principali per spiegare e prevedere
i dati basandosi sui loro valori passati e sugli errori
di previsione passati:
il componente Autoregressivo (AR),
la Differenziazione (I - Integrated) per rendere la serie stazionaria,
e il componente a Media Mobile (MA).

Viene specificato da tre parametri:
    p (ordine AR),
    d (ordine di differenziazione)
    q (ordine MA)

SARIMA
Seasonal Autoregressive Integrated Moving Average
è un'estensione del modello ARIMA che aggiunge componenti stagionali per prevedere
serie temporali con pattern periodici ricorrenti, come vendite mensili
o temperature annuali, introducendo parametri specifici (P, D, Q, m)
per catturare queste fluttuazioni stagionali,
oltre ai parametri standard (p, d, q) del modello ARIMA non stagionale,
rappresentando il tutto nella notazione SARIMA(p, d, q)(P, D, Q)m.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima #Useremo pmdarima per trovare il modello migliore
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# 1. Creazione di un dataset CSV fittizio (simulazione vendite mensili per 3 anni)
def create_dummy_csv():
    dates = pd.date_range(start="2022-01-01", end="2025-01-01", freq='MS') # Month Start frequency
    # Crea vendite con un trend crescente e un po' di stagionalità
    sales = np.linspace(100, 300, len(dates)) + np.random.normal(0, 15, len(dates)) + 20 * np.sin(np.arange(len(dates)) * np.pi / 6)
    df = pd.DataFrame({'Data': dates, 'Vendite': sales})
    df['Data'] = df['Data'].dt.strftime('%Y-%m-%d') # Formato compatibile CSV
    df.to_csv('vendite_mensili.csv', index=False)
    print("Creato il file 'vendite_mensili.csv'")

create_dummy_csv()

# 2. Caricamento e preparazione del dataset
df = pd.read_csv('vendite_mensili.csv', parse_dates=['Data'], index_col='Data')

print("\nPrime 5 righe del dataset:")
print(df.head())
df['Vendite'].plot(title="Vendite Mensili Storiche")
plt.show()

# 3. Suddivisione in Training e Test set
# Addestriamo su tutti i dati tranne gli ultimi 4 mesi, che usiamo per testare
train_data = df[:-4]
test_data = df[-4:]

print(f"\nDati di addestramento: {len(train_data)} mesi")
print(f"Dati di test: {len(test_data)} mesi")

# 4. Trovare i parametri ARIMA ottimali automaticamente con pmdarima
# auto_arima cerca la migliore combinazione di (p, d, q) e parametri stagionali (P, D, Q, S)
print("\nRicerca parametri ARIMA ottimali (potrebbe richiedere qualche secondo)...")
model_auto_arima = auto_arima(train_data['Vendite'], 
                              start_p=0, start_q=0,
                              test='adf',       # Usa Adfuller test per trovare d
                              max_p=3, max_q=3, # Max p e q
                              m=12,             # Frequenza stagionale (12 mesi)
                              d=None,           # Let the algorithm determine d
                              seasonal=True,    # Considera la stagionalità
                              start_P=0, 
                              D=1,              # Spesso D=1 è sufficiente per dati mensili
                              trace=True,
                              error_action='ignore',  
                              suppress_warnings=True, 
                              stepwise=True)

print(f"\nModello ARIMA/SARIMA ottimale trovato: {model_auto_arima.summary()}")

# 5. Eseguire la previsione (Forecasting)
n_periods = len(test_data) # Prevediamo per la stessa durata del test set
fitted, confint = model_auto_arima.predict(n_periods=n_periods, return_conf_int=True)
forecast_index = test_data.index

# Crea una serie pandas per la previsione
forecast_series = pd.Series(fitted, index=forecast_index)
lower_series = pd.Series(confint[:, 0], index=forecast_index)
upper_series = pd.Series(confint[:, 1], index=forecast_index)

# Calcolo dell'errore (MSE)
mse = mean_squared_error(test_data['Vendite'], forecast_series)
print(f"\nErrore Quadratico Medio (MSE) sulle previsioni: {mse:.2f}")

# 6. Visualizzazione dei risultati
plt.figure(figsize=(12, 6))
plt.plot(train_data.index, train_data['Vendite'], label='Dati di Addestramento')
plt.plot(test_data.index, test_data['Vendite'], color='green', label='Valori Reali (Test)')
plt.plot(forecast_series.index, forecast_series, color='darkorange', label='Previsione ARIMA')
plt.fill_between(lower_series.index, 
                 lower_series, 
                 upper_series, 
                 color='k', alpha=.15, label='Intervallo di Confidenza 95%')

plt.title("Previsione Vendite Mensili con Modello ARIMA/SARIMA")
plt.xlabel("Data")
plt.ylabel("Vendite")
plt.legend()
plt.show()
