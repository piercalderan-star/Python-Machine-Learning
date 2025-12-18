import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("vendite_mensili.csv")

# Parsing della colonna data
df["Data"] = pd.to_datetime(df["Data"])
df = df.set_index("Data").sort_index()

print(df.head())
print(df.index.freq)  # può essere None, la si può impostare a mano

# Plot base
df["Vendite"].plot(marker="o")
plt.title("Vendite mensili")
plt.xlabel("Data")
plt.ylabel("Vendite")
plt.grid(True)
plt.show()
