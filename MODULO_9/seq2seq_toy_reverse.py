'''
Mini seq2seq “toy translator” (reverse string)

File: seq2seq_toy_reverse.py
(L’idea è didattica: far vedere encoder/decoder, non fare vera traduzione.)
'''
import torch
from torch import nn, optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Toy dataset: dato "abc", deve produrre "cba"
pairs = [
    ("ciao", "oaic"),
    ("mare", "eram"),
    ("pippo", "oppip"),
    ("test", "tset")
]

# Creiamo un vocab caratteri
chars = set()
for src, tgt in pairs:
    chars.update(list(src))
    chars.update(list(tgt))
chars = sorted(list(chars))
stoi = {"<pad>": 0, "<sos>": 1, "<eos>": 2}
for c in chars:
    stoi[c] = len(stoi)
itos = {i: s for s, i in stoi.items()}

vocab_size = len(stoi)
max_len = max(len(s) for s, _ in pairs) + 2  # + sos + eos

def encode_seq(s, add_sos_eos=False):
    tokens = []
    if add_sos_eos:
        tokens.append(stoi["<sos>"])
    for ch in s:
        tokens.append(stoi[ch])
    if add_sos_eos:
        tokens.append(stoi["<eos>"])
    # pad
    tokens = tokens[:max_len]
    tokens += [stoi["<pad>"]] * (max_len - len(tokens))
    return torch.tensor(tokens, dtype=torch.long)

data = []
for src, tgt in pairs:
    data.append((encode_seq(src, add_sos_eos=False),
                 encode_seq(tgt, add_sos_eos=True)))

class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=32, hidden_dim=64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.rnn = nn.GRU(emb_dim, hidden_dim, batch_first=True)

    def forward(self, src):
        emb = self.emb(src)
        _, h = self.rnn(emb)
        return h  # ultimo hidden

class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=32, hidden_dim=64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.rnn = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tgt, hidden):
        # tgt: [batch=1, seq_len]
        emb = self.emb(tgt)
        out, h = self.rnn(emb, hidden)
        logits = self.fc(out)
        return logits, h

enc = Encoder(vocab_size).to(device)
dec = Decoder(vocab_size).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=stoi["<pad>"])
params = list(enc.parameters()) + list(dec.parameters())
optimizer = optim.Adam(params, lr=1e-3)

def train_epoch():
    enc.train(); dec.train()
    total_loss = 0
    for src, tgt in data:
        src = src.unsqueeze(0).to(device)  # batch=1
        tgt = tgt.unsqueeze(0).to(device)

        optimizer.zero_grad()
        h = enc(src)  # [1, batch, hidden] -> [1,1,hidden]

        # teacher forcing
        logits, _ = dec(tgt[:, :-1], h)   # predice tutti i token eccetto l'ultimo
        # shift targhet di 1 a destra
        target = tgt[:, 1:].contiguous()

        loss = criterion(logits.view(-1, vocab_size), target.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data)

for epoch in range(200):
    loss = train_epoch()
    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1}, loss: {loss:.4f}")

# Funzione di decoding
def decode(src_str):
    enc.eval(); dec.eval()
    src = encode_seq(src_str, add_sos_eos=False).unsqueeze(0).to(device)
    with torch.no_grad():
        h = enc(src)

        # Decodifica step-by-step
        inp = torch.tensor([[stoi["<sos>"]]], dtype=torch.long, device=device)
        out_chars = []
        for _ in range(max_len):
            logits, h = dec(inp, h)
            # prendiamo l'ultimo timestep
            next_token = logits[:, -1, :].argmax(dim=-1)
            token_id = next_token.item()
            if token_id == stoi["<eos>"]:
                break
            if token_id != stoi["<pad>"]:
                out_chars.append(itos[token_id])
            inp = torch.cat([inp, next_token.unsqueeze(0)], dim=1)
    return "".join(out_chars)

print("Test decoding:")
for src, _ in pairs:
    print(src, "->", decode(src))
