'''
L'analisi del sentimento è un problema di NLP (Natural Language Processing)
e richiede un approccio leggermente diverso rispetto alla regressione numerica.

Per gestire il testo con PyTorch, dobbiamo prima:
Tokenizzare il testo (suddividerlo in parole/numeri).
Vettorizzare il testo (convertire le parole in numeri comprensibili per la rete neurale).
Usare un'architettura di rete neurale adatta a sequenze,
come un semplice embedding layer seguito da un layer ricorrente (RNN/LSTM)
    RNN (Reti Neurali Ricorrenti) sono modelli dati sequenziali,
    LSTM (Long Short-Term Memory) sono un tipo avanzato di RNN,
    specificamente progettate per risolvere il problema
    delle dipendenze a lungo termine
    
In questo caso, per semplicità, si usa un semplice feed-forward.
Questo esempio è più complesso del precedente a causa della natura del dato (testo),
ma mostra l'uso di PyTorch per la classificazione binaria (positivo/negativo = 0 e 1).

Esempio: Analisi del Sentimento del cliente con PyTorch
    Questo esempio include la pre-elaborazione del testo manuale
    per mantenere la dipendenza da librerie esterne al minimo,
    concentrandosi su sklearn, pandas, torch, e matplotlib.
'''

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
import torch.nn.functional as F

# 1. Preparazione del database fittizio
data = {
    'descrizione': [
        "terribile qualità", "fantastico prodotto", "prodotto mediocre", "eccellente, lo adoro",
        "pessima esperienza", "buono, ma caro", "non male", "soldi buttati",
        "super consigliato", "rotto dopo un giorno", "funziona perfettamente",
        "qualità scarsa", "assistenza clienti ottima", "lento e inutile",
        "vale il prezzo", "mai più", "incredibile acquisto"
    ],
    'etichetta': [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1] # 0=Negativo, 1=Positivo
}
df = pd.DataFrame(data)

# 2. Pre-elaborazione del testo e Vettorizzazione (Bag of Words)
# Convertiamo il testo in una rappresentazione numerica usando CountVectorizer
# che crea un vocabolario e conta le occorrenze delle parole.
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['descrizione']).toarray() # X è una matrice di conteggi parole
y = df['etichetta'].values.reshape(-1, 1)

# Dividi i dati in set di addestramento e test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Converti in Tensori PyTorch
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)

# 3. Definizione del Modello di Rete Neurale
# L'input_dim è la dimensione del vocabolario creato dal vectorizer
input_dim = X_train_t.shape[1]

class SentimentClassifier(nn.Module):
    def __init__(self, input_dim):
        super(SentimentClassifier, self).__init__()
        # Un semplice layer lineare seguito da Sigmoid per la classificazione binaria
        self.fc1 = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = self.fc1(x)
        # La funzione sigmoide "schiaccia" l'output tra 0 e 1 (probabilità)
        return torch.sigmoid(x)

model = SentimentClassifier(input_dim)

# 4. Definizione di Loss e Ottimizzatore
# Binary Cross-Entropy Loss è standard per la classificazione binaria
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 5. Ciclo di Addestramento (Training Loop)
num_epochs = 100
loss_history = []

for epoch in range(num_epochs):
    y_pred = model(X_train_t)
    loss = criterion(y_pred, y_train_t)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    loss_history.append(loss.item())

    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 6. Visualizzazione della curva di perdita con Matplotlib
plt.figure(figsize=(10, 5))
plt.plot(loss_history)
plt.title('Curva di Perdita (Training Loss) di PyTorch - Analisi Sentimento')
plt.xlabel('Epoche')
plt.ylabel('Loss (Binary Cross-Entropy)')
plt.grid(True)
plt.show()

# 7. Valutazione e Previsione (Opzionale)
model.eval() 
with torch.no_grad():
    test_predictions_t = model(X_test_t)
    # Converti le probabilità in classi ( > 0.5 è 1/Positivo, altrimenti 0/Negativo)
    test_predictions = (test_predictions_t > 0.5).int().numpy().flatten()
    
    print(f"\nAccuratezza sul set di test: {accuracy_score(y_test.flatten(), test_predictions):.2f}")
    print("\nPrevisioni vs Valori Reali (Test Set):")
    for i in range(len(test_predictions)):
        sentimento = "Positivo" if test_predictions[i] == 1 else "Negativo"
        reale = "Positivo" if y_test.flatten()[i] == 1 else "Negativo"
        # Mostra la descrizione originale
        original_desc = df.iloc[y_test_t.shape[0] + i]['descrizione']
        print(f"Descrizione: '{original_desc}' | Previsto: {sentimento}, Reale: {reale}")



