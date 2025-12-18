'''
https://pandas.pydata.org/

per test online:
https://pandas.pydata.org/try.html

https://matplotlib.org/
esempi:
https://matplotlib.org/stable/plot_types/index.html
'''

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("vendite_2024.csv")

df["prezzo"].plot(kind="hist")
plt.title("Distribuzione prezzi")
plt.show()

df.groupby("mese")["prezzo"].sum().plot(kind="bar")
plt.title("Vendite mensili")
plt.show()
