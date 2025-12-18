'''
L'uso di una GRU (Gated Recurrent Unit)
per la classificazione del testo è un ottimo passo avanti nel deep learning
rispetto al semplice CountVectorizer visto in precedenza.

Le GRU, come le LSTM, sono progettate per gestire le dipendenze nelle sequenze
(come l'ordine delle parole in una frase).
 
In questo esempio useremo lo stesso dataset di sentimenti
("terribile qualità", negativo, ecc.),
ma useremo un approccio più sofisticato:

Tokenizzazione Keras:
 Useremo il tokenizer di Keras
 (una libreria di alto livello per il deep learning) per preparare il testo.

PyTorch GRU:
 Costruiremo il modello di rete neurale usando PyTorch con un Embedding Layer e un GRU Layer.
'''

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

# 1. Preparazione del database fittizio
data = {
    'descrizione': [
        "terribile qualità",
        "fantastico prodotto",
        "prodotto mediocre",
        "eccellente, lo adoro",
        "pessima esperienza",
        "buono, ma caro",
        "non male",
        "soldi buttati",
        "super consigliato",
        "rotto dopo un giorno",
        "funziona perfettamente",
        "qualità scarsa",
        "assistenza clienti ottima",
        "lento e inutile",
        "vale il prezzo",
        "mai più",
        "incredibile acquisto",
        "pessimo",
        "stupendo"
    ],
    'etichetta': [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0,1] # 0=Negativo, 1=Positivo
}

df = pd.DataFrame(data)

# 2. Tokenizzazione e Vettorizzazione Keras/Pad_sequences
# Questa parte trasforma le parole in indici numerici e assicura che tutte le frasi abbiano la stessa lunghezza
tokenizer = Tokenizer(num_words=1000, oov_token="<UNK>") # Limita a 1000 parole uniche
tokenizer.fit_on_texts(df['descrizione'])

X_sequences = tokenizer.texts_to_sequences(df['descrizione'])

# Padding: rende tutte le sequenze lunghe uguali (importante per le reti neurali)
MAX_LEN = 5
X_padded = pad_sequences(X_sequences, maxlen=MAX_LEN, padding='post', truncating='post')

y = df['etichetta'].values.reshape(-1, 1)

# Suddivisione in Training e Test set
X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
    X_padded, y, test_size=0.3, random_state=42
)

# Conversione in Tensori PyTorch (LongTensor per gli indici delle parole)
X_train_t = torch.tensor(X_train_np, dtype=torch.long)
y_train_t = torch.tensor(y_train_np, dtype=torch.float32)
X_test_t = torch.tensor(X_test_np, dtype=torch.long)
y_test_t = torch.tensor(y_test_np, dtype=torch.float32)

VOCAB_SIZE = len(tokenizer.word_index) + 1 # Dimensione vocabolario + 1 per UNK

# 3. Definizione del Modello PyTorch con GRU
class SentimentGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        # Layer 1: Embedding - trasforma gli indici numerici in vettori densi (significato)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Layer 2: GRU - processa la sequenza di embeddings
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        
        # Layer 3: Fully Connected (Lineare) - trasforma l'output GRU in previsione finale
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        # out contiene l'output per ogni time step
        # hidden contiene lo stato finale
        out, hidden = self.gru(embedded) 
        # Passiamo solo l'ultimo stato hidden al layer lineare
        return self.fc(hidden.squeeze(0)) # Squeeze rimuove la dimensione batch per il layer lineare

EMBEDDING_DIM = 100
HIDDEN_DIM = 64
OUTPUT_DIM = 1 # 1 output per la classificazione binaria (0 o 1)

model = SentimentGRU(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)

# 4. Definizione di Loss e Ottimizzatore
criterion = nn.BCEWithLogitsLoss() # Include Sigmoid + Binary Cross Entropy (più stabile)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. Ciclo di Addestramento (Training Loop)
num_epochs = 50
loss_history = []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    predictions = model(X_train_t)
    loss = criterion(predictions, y_train_t)
    
    loss.backward()
    optimizer.step()
    
    loss_history.append(loss.item())

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 6. Valutazione e Previsione
model.eval() 
with torch.no_grad():
    test_predictions = model(X_test_t)
    # Applica Sigmoid e converte in classi ( > 0.5 è 1/Positivo)
    test_predictions_class = torch.sigmoid(test_predictions)
    test_predictions_class = (test_predictions_class > 0.5).int().numpy().flatten()
    
    accuracy = (test_predictions_class == y_test_np.flatten()).mean()
    print(f"\nAccuratezza sul set di test: {accuracy:.2f}")

# 7. Funzione di input utente per previsione singola
def predict_sentiment_gru(text_input, model, tokenizer, max_len):
    model.eval()
    # Prepara il testo dell'utente come fatto per il training
    seq = tokenizer.texts_to_sequences([text_input])
    padded = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
    input_t = torch.tensor(padded, dtype=torch.long)
    
    with torch.no_grad():
        output = model(input_t)
        # Ottieni la probabilità finale
        probability = torch.sigmoid(output).item()
        
    sentiment = "Positivo" if probability > 0.5 else "Negativo"
    return sentiment, probability

# Test interattivo
test_phrase = "il prodotto è ok"
sentiment, prob = predict_sentiment_gru(test_phrase, model, tokenizer, MAX_LEN)
print(f"\nFrase: '{test_phrase}'")
print(f"Previsione: {sentiment} (Probabilità Positiva: {prob:.4f})")

test_phrase_2 = "qualità terribile e non funziona"
sentiment_2, prob_2 = predict_sentiment_gru(test_phrase_2, model, tokenizer, MAX_LEN)
print(f"\nFrase: '{test_phrase_2}'")
print(f"Previsione: {sentiment_2} (Probabilità Positiva: {prob_2:.4f})")
