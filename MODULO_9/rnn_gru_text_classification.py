import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

# Dataset di frasi molto semplice, giusto per demo
sentences = [
    "oggi è una bella giornata",
    "odio quando piove tutto il giorno",
    "adoro studiare machine learning",
    "non mi piace il traffico in città"
]
labels = [1, 0, 1, 0]  # 1=positivo, 0=negativo

# Costruiamo un vocabolario semplice
word2idx = {"<pad>": 0, "<unk>": 1}
for s in sentences:
    for w in s.split():
        if w not in word2idx:
            word2idx[w] = len(word2idx)

vocab_size = len(word2idx)
print("Vocab size:", vocab_size)

def encode_sentence(s, max_len=8):
    tokens = [word2idx.get(w, 1) for w in s.split()]
    tokens = tokens[:max_len]
    tokens += [0] * (max_len - len(tokens))
    return tokens

X = [encode_sentence(s) for s in sentences]
y = labels

class SimpleTextDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

ds = SimpleTextDataset(X, y)
loader = DataLoader(ds, batch_size=2, shuffle=True)

# Modello GRU
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim=32, hidden_dim=64, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        emb = self.embedding(x)      # [batch, seq, emb_dim]
        out, h_n = self.gru(emb)     # out: [batch, seq, hidden]
        # Usiamo l'ultimo hidden state come rappresentazione globale
        last_hidden = h_n[-1]        # [batch, hidden_dim]
        logits = self.fc(last_hidden)
        return logits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GRUClassifier(vocab_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, loss medio: {total_loss/len(loader):.4f}")

# Test rapido
model.eval()
test_sent = "amo il bel tempo"
x = torch.tensor([encode_sentence(test_sent)], dtype=torch.long).to(device)
with torch.no_grad():
    logits = model(x)
    pred = logits.argmax(dim=1).item()
print(f"Frase: {test_sent}")
print("Predizione (1=positivo,0=negativo):", pred)
