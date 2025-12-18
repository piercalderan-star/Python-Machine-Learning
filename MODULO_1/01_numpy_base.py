'''

prova online il codice

API array
https://numpy.org/doc/stable/reference/arrays.html
'''

import numpy as np

# Creazione array
a = np.array([1, 2, 3])
b = np.arange(0, 10, 2)
c = np.linspace(0, 1, 5)

print("Array a:", a)
print("Array b:", b)
print("Array c:", c)

print()

# Operazioni matematiche
print("a * 10 =", a * 10)
print("a + b[:3] =", a + b[:3])

print()

# Statistiche
print("Media di b:", b.mean())
print("Somma di b:", b.sum())

print()

# Reshape
m = np.arange(1, 10).reshape(3, 3)
print("Matrice 3x3:\n", m)
