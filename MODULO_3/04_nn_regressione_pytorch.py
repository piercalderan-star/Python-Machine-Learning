'''
PyTorch è una libreria open source
per l'apprendimento automatico (machine learning)
e l'apprendimento profondo (deep learning),
sviluppata originariamente da Meta AI (Facebook AI Research, o FAIR). 

In breve:
Libreria Python:
    È costruita principalmente per l'uso con il linguaggio
    di programmazione Python.

Tensor Flow (Simile a NumPy):
    Il suo elemento fondamentale sono i tensori
    (oggetti simili agli array multidimensionali di NumPy)
    ottimizzati per l'uso su GPU (schede grafiche),
    permettendo calcoli molto veloci.

Grafi Computazionali Dinamici:
    A differenza di altri framework come TensorFlow
    (che in passato usava grafi statici),
    PyTorch utilizza grafi computazionali dinamici.
    Questo rende la definizione e il debugging delle reti neurali
    molto più flessibile e intuitiva per i programmatori.

Flessibilità e Ricerca:
    È estremamente apprezzato nella comunità della ricerca accademica
    e tra i professionisti per la sua flessibilità
    e facilità di prototipazione.
    
È uno dei due framework principali
(l'altro è Google TensorFlow) utilizzati a livello
globale per creare e addestrare modelli
di intelligenza artificiale avanzati.

Perché non funziona TensorFlow?
TensorFlow (anche 2.13 / 2.14 / 2.15) supporta solo:
Python	Supporto TF
3.7	✔
3.8	✔
3.9	✔
3.10	✔
3.11	❌ no ufficiale
3.12	❌ no ufficiale
3.13	❌ impossibile
'''

#Esempio 1 Addestramento, Perdita (Loss) e regressione
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 1. Preparazione del database fittizio
# quartiere 1 = centro
# quartiere 2 = periferia

data = {
    'Superficie': [45, 70, 44, 55, 65, 35, 85, 90, 120, 100],
    'Stanze': [2, 3, 2, 2, 2, 1, 3, 4, 4, 3],
    'Quartiere': [2, 1, 2, 2, 2, 1, 2, 1, 2, 2], # 1 o 2
    'Prezzo': [78000, 120000, 80000, 62000, 56000, 33000, 110000, 120000, 200000, 98000]
}
df = pd.DataFrame(data)

# Definisce le feature (X) e il target (y)
features = ['Superficie', 'Stanze', 'Quartiere']
X = df[features].values
y = df['Prezzo'].values.reshape(-1, 1) # Assicurati che y sia una colonna singola

# 2. Pre-elaborazione e Standardizzazione dei Dati
# La standardizzazione è cruciale per le reti neurali
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y)

# Divide i dati in set di addestramento e test
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(
    X_scaled, y_scaled, test_size=0.3, random_state=42
)

# 3. Conversione in Tensori PyTorch
# I dati devono essere in formato float32 per PyTorch
X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train_scaled, dtype=torch.float32)
X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_t = torch.tensor(y_test_scaled, dtype=torch.float32)

# 4. Definizione del Modello di Rete Neurale
# Il nostro modello avrà 3 input (feature) e 1 output (prezzo)
class PricePredictor(nn.Module):
    def __init__(self):
        super(PricePredictor, self).__init__()
        # Un singolo layer lineare (regressione lineare pura)
        self.fc1 = nn.Linear(in_features=3, out_features=1)

    def forward(self, x):
        x = self.fc1(x)
        return x

model = PricePredictor()

# 5. Definizione della Funzione di Perdita (Loss) e dell'Ottimizzatore
# Usiamo l'errore quadratico medio (MSE = Medium Square Error))
# per la regressione

criterion = nn.MSELoss()

# L'ottimizzatore Adam aggiorna i pesi del modello
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 6. Ciclo di Addestramento (Training Loop)
num_epochs = 500
loss_history = []

for epoch in range(num_epochs):
    # Forward pass: calcola la previsione
    y_pred = model(X_train_t)
    
    # Calcola la perdita (Loss)
    loss = criterion(y_pred, y_train_t)
    
    # Backward pass e ottimizzazione: azzera i gradienti, calcola i gradienti, aggiorna i pesi
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Salva la perdita per la visualizzazione
    loss_history.append(loss.item())

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 7. Visualizzazione della curva di perdita con Matplotlib
plt.figure(figsize=(10, 5))
plt.plot(loss_history)
plt.title('Curva di Perdita (Training Loss) di PyTorch')
plt.xlabel('Epoche')
plt.ylabel('Loss (MSE)')
plt.grid(True)
plt.show()

# 8. Valutazione e Previsione (Opzionale)
model.eval() # Imposta il modello in modalità valutazione

with torch.no_grad():
    test_predictions_scaled = model(X_test_t)
    test_loss = criterion(test_predictions_scaled, y_test_t)
    
    # Riconverte i risultati alla scala originale dei prezzi (in €)
    test_predictions = scaler_y.inverse_transform(test_predictions_scaled.numpy())
    y_test_original = scaler_y.inverse_transform(y_test_scaled)
    
    print(f"\nPerdita sul set di test (scaled): {test_loss.item():.4f}")
    print("\nPrevisioni vs Valori Reali (Esempio Test Set):")
    
    for i in range(len(test_predictions)):
        print(f"Previsto: {test_predictions[i][0]:.2f} €, Reale: {y_test_original[i][0]:.2f} €")



#Esempio 2: Addestramento da CSV e input utente
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# --- PARTE 1: Preparazione e Addestramento (COME PRIMA) ---
data = {
    'Superficie': [45, 70, 44, 55, 65, 35, 85, 90, 120, 100],
    'Stanze': [2, 3, 2, 2, 2, 1, 3, 4, 4, 3],
    'Quartiere': [2, 1, 2, 2, 2, 1, 2, 1, 2, 2], # 1 o 2
    'Prezzo': [78000, 120000, 80000, 62000, 56000, 33000, 110000, 120000, 200000, 98000]
}

df = pd.read_csv("prezzi_case_nn.csv")

X = df[["superficie", "stanze", "quartiere"]].values.astype("float32")
y = df["prezzo"].values.astype("float32").reshape(-1, 1)


scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y)

X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(
    X_scaled, y_scaled, test_size=0.3, random_state=42
)

X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train_scaled, dtype=torch.float32)

class PricePredictor(nn.Module):
    def __init__(self):
        super(PricePredictor, self).__init__()
        self.fc1 = nn.Linear(in_features=3, out_features=1)
    def forward(self, x):
        x = self.fc1(x)
        return x

model = PricePredictor()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 500
for epoch in range(num_epochs):
    y_pred = model(X_train_t)
    loss = criterion(y_pred, y_train_t)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Addestramento completato.")
# --- FINE PARTE 1 ---

# --- PARTE 2: Input Utente e Predizione ---

print("\n--- Previsione Prezzo Casa ---")
try:
    # Richiesta input all'utente
    superficie = float(input("Inserisci la superficie (mq): "))
    stanze = float(input("Inserisci il numero di stanze: "))
    quartiere = float(input("Inserisci il quartiere (1=Centro, 2=Periferia): "))

    # Prepara i dati inseriti dall'utente in un formato che il modello capisca
    # Deve essere un array 2D: [[superficie, stanze, quartiere]]
    user_data = np.array([[superficie, stanze, quartiere]])

    # Standardizza i dati dell'utente usando lo STESSO scaler usato per l'addestramento
    user_data_scaled = scaler_X.transform(user_data)
    
    # Converte in tensore PyTorch
    user_data_t = torch.tensor(user_data_scaled, dtype=torch.float32)

    # Esegue la previsione (metti il modello in modalità valutazione)
    model.eval()
    with torch.no_grad():
        prediction_scaled = model(user_data_t)
        
        # Riconverti il risultato alla scala originale dei prezzi (in €)
        prediction_original = scaler_y.inverse_transform(prediction_scaled.numpy())
        
        # Stampa il risultato finale
        print(f"\nRisultato della previsione:")
        print(f"Il prezzo previsto per la casa è di circa: {prediction_original[0][0]:.2f} €")

except ValueError:
    print("Errore nell'input. Assicurati di inserire valori numerici validi.")
except Exception as e:
    print(f"Si è verificato un errore: {e}")

# La visualizzazione della loss curve dell'addestramento è omessa qui per brevità



