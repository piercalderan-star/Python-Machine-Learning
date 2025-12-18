"""
Usare il modello per predizioni reali
"""

import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("prezzi_case_ml.csv")

model = LinearRegression()
model.fit(df[["superficie"]], df["prezzo"])

superficie_nuova = 90
prezzo_predetto = model.predict([[superficie_nuova]])

print(f"Per {superficie_nuova} m2 il prezzo stimato è: {prezzo_predetto[0]:.2f} €")


