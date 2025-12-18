import pandas as pd

df = pd.read_csv("vendite_2024.csv")

print(df.head())
print()
print(df.info())
print()
print(df.describe())
print()

# Selezione colonne
print(df["categoria"].unique())

print()

# Filtri
print(df[df["prezzo"] > 50])

print()

# Nuova colonna
df["iva"] = df["prezzo"] * 0.22

print()

# Gestione valori NaN
df["prezzo"] = df["prezzo"].fillna(df["prezzo"].mean())

print()

# GroupBy
g = df.groupby("categoria")["prezzo"].mean()
print(g)


