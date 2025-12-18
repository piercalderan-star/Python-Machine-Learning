import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

df = pd.read_csv("vendite_mensili.csv")
df["Data"] = pd.to_datetime(df["Data"])
df = df.set_index("Data").sort_index()

# Train = primi 80%, Test = ultimi 20%
split_idx = int(len(df) * 0.8)
train = df.iloc[:split_idx]
test = df.iloc[split_idx:]

y_train = train["Vendite"]
y_test = test["Vendite"]

# Baseline 1: Naive (previsione = Vendite precedente)
y_pred_naive = y_test.shift(1)  # attenzione, bisogna allineare bene

# Per fare naive bene, usiamo l'ultimo Vendite del train e poi shiftiamo
shifted = df["Vendite"].shift(1)
y_pred_naive = shifted.iloc[split_idx:]

# Baseline 2: media mobile (finestra 3)
rolling = df["Vendite"].rolling(window=3).mean()
y_pred_ma = rolling.iloc[split_idx:]

def metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse

mae_naive, rmse_naive = metrics(y_test, y_pred_naive)
mae_ma, rmse_ma = metrics(y_test, y_pred_ma)

print("Baseline Naive: MAE =", mae_naive, "RMSE =", rmse_naive)
print("Baseline Media Mobile: MAE =", mae_ma, "RMSE =", rmse_ma)
