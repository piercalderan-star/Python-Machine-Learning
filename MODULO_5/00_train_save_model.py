# train_save_model.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("prezzi_case_ml.csv")

X = df[["superficie", "stanze", "zona"]]
y = df["prezzo"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

print("Score:", model.score(X_test, y_test))

# Salvataggio modello
joblib.dump(model, "modello_case.pkl")
print("Modello salvato come modello_case.pkl")
