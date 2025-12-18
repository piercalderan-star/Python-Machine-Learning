"""
Progetto B – Previsione di serie temporali con GRU

- Carica un dataset di temperatura sintetica (timeseries_temp.csv)
- Crea finestre (sequence -> next_value)
- Addestra una GRU per predire il valore futuro
- Mostra un confronto tra valori reali e predetti

Prerequisiti:
- pip install torch pandas matplotlib
"""

import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

CSV_PATH = "timeseries_temp.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TimeSeriesDataset(Dataset):
    def __init__(self, series, window_size=20):
        self.series = series
        self.window_size = window_size

    def __len__(self):
        return len(self.series) - self.window_size

    def __getitem__(self, idx):
        x = self.series[idx:idx+self.window_size]
        y = self.series[idx+self.window_size]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class GRURegressor(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: [batch, seq, 1]
        out, h = self.gru(x)
        last = out[:, -1, :]
        y = self.fc(last)
        return y.squeeze(-1)

def main():
    df = pd.read_csv(CSV_PATH)
    values = df["value"].values.astype("float32")
    train_size = int(0.8 * len(values))
    train_vals = values[:train_size]
    test_vals = values[train_size-20:]  # includo finestra

    window_size = 20
    train_ds = TimeSeriesDataset(train_vals, window_size=window_size)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    model = GRURegressor().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print("Inizio training...")
    for epoch in range(20):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.unsqueeze(-1).to(DEVICE)  # [B,seq,1]
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoca {epoch+1}, loss medio: {total_loss/len(train_loader):.4f}")

    # Predizione su test
    model.eval()
    preds = []
    with torch.no_grad():
        seq = test_vals[:window_size].copy()
        for i in range(len(test_vals) - window_size):
            inp = torch.tensor(seq[-window_size:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEVICE)
            pred = model(inp).item()
            preds.append(pred)
            seq = list(seq) + [pred]

    real = test_vals[window_size:]
    plt.figure(figsize=(10,4))
    plt.plot(real, label="Reale")
    plt.plot(preds, label="Predetto")
    plt.legend()
    plt.title("Progetto B – Previsione serie temporali (GRU)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
