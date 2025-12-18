import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

# 1. Creazione di un dataset CSV fittizio (temperature giornaliere per 2 anni)


def create_temp_csv():
    dates = pd.date_range(start="2023-01-01", end="2025-01-01", freq='D')
    
    # Simula temperature con stagionalità e rumore
    temperatures = 15 + 10 * np.sin(np.arange(len(dates)) * 2 * np.pi / 365) + np.random.normal(0, 1.5, len(dates))
    
    df = pd.DataFrame({'Data': dates, 'Temperatura': temperatures})
    
    df.to_csv('temperature_giornaliere.csv', index=False)
    
    print("Creato il file 'temperature_giornaliere.csv'")

create_temp_csv()

# 2. Caricamento e Preparazione Dati per LSTM
df = pd.read_csv('temperature_giornaliere.csv', parse_dates=['Data'], index_col='Data')

data = df['Temperatura'].values.astype(float)

# Normalizzazione dei dati (essenziale per reti neurali)
scaler = MinMaxScaler(feature_range=(-1, 1))
data_normalized = scaler.fit_transform(data.reshape(-1, 1))

# Funzione per creare sequenze (input_sequence_length = X giorni precedenti)
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length] # Prevede il giorno successivo
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

SEQUENCE_LENGTH = 30 # Usiamo gli ultimi 30 giorni per prevedere il prossimo
X_np, y_np = create_sequences(data_normalized, SEQUENCE_LENGTH)

# Suddivisione in Training (80%) e Test (20%)
train_size = int(len(X_np) * 0.8)
X_train_np, X_test_np = X_np[:train_size], X_np[train_size:]
y_train_np, y_test_np = y_np[:train_size], y_np[train_size:]

# Conversione in Tensori PyTorch
X_train_t = torch.tensor(X_train_np, dtype=torch.float32)
y_train_t = torch.tensor(y_train_np, dtype=torch.float32)
X_test_t = torch.tensor(X_test_np, dtype=torch.float32)
y_test_t = torch.tensor(y_test_np, dtype=torch.float32)

# 3. Definizione del Modello PyTorch LSTM
class TempLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(TempLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Batch_first=True significa che l'input ha la forma (batch, seq, feature)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Inizializza hidden e cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        
        # Forward propagate LSTM
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        
        # Decodifica l'output dell'ultimo time step
        out = self.fc(out[:, -1, :])
        return out

INPUT_DIM = 1 # Una sola feature (Temperatura)
HIDDEN_DIM = 32
OUTPUT_DIM = 1
NUM_LAYERS = 1

model = TempLSTM(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS)
criterion = nn.MSELoss() # Errore quadratico medio per regressione
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Addestramento del Modello
num_epochs = 100
print("\nInizio addestramento LSTM...")

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    y_pred = model(X_train_t)
    loss = criterion(y_pred, y_train_t)
    
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Addestramento terminato.")

# 5. Previsione e Visualizzazione
model.eval()
with torch.no_grad():
    # Previsioni sia sul training set che sul test set per visualizzazione
    train_predict = model(X_train_t)
    test_predict = model(X_test_t)

# Riconverti i dati scalati alla scala originale delle temperature (°C)
train_predict_rescaled = scaler.inverse_transform(train_predict.numpy())
y_train_rescaled = scaler.inverse_transform(y_train_t.numpy())
test_predict_rescaled = scaler.inverse_transform(test_predict.numpy())
y_test_rescaled = scaler.inverse_transform(y_test_t.numpy())

# Preparazione dei dati per il grafico (allinea le previsioni con le date corrette)
train_dates = df.index[SEQUENCE_LENGTH:train_size + SEQUENCE_LENGTH]
test_dates = df.index[train_size + SEQUENCE_LENGTH:]

plt.figure(figsize=(15, 6))
plt.plot(df.index, data, label='Dati Reali Completi', color='gray', alpha=0.5)
plt.plot(train_dates, train_predict_rescaled, label='Previsioni Training (LSTM)', color='blue')
plt.plot(test_dates, test_predict_rescaled, label='Previsioni Test (LSTM)', color='orange')
plt.title("Previsione Temperature Giornaliere con LSTM")
plt.xlabel("Data")
plt.ylabel("Temperatura (°C)")
plt.legend()
plt.show()















