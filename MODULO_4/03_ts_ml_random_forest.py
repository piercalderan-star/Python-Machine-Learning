import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv("vendite_mensili.csv")
df["Data"] = pd.to_datetime(df["Data"])
df = df.set_index("Data").sort_index()

# Creazione feature lag (Vendite t-1, t-2, t-3) e mese
df["lag1"] = df["Vendite"].shift(1)
df["lag2"] = df["Vendite"].shift(2)
df["lag3"] = df["Vendite"].shift(3)
df["mese"] = df.index.month

df = df.dropna()  # per togliere le prime righe senza lag completi

feature_cols = ["lag1", "lag2", "lag3", "mese"]
X = df[feature_cols]
y = df["Vendite"]

# Train/Test temporale
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("RandomForest ML: MAE =", mae, "RMSE =", rmse)

# Mostriamo qualche confronto
res = pd.DataFrame({"reale": y_test, "predetto": y_pred}, index=y_test.index)
print(res.head())
