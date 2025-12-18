'''
La Convolutional Neural Network (CNN o ConvNet)
è un tipo specializzato di rete neurale artificiale,
usata prevalentemente ed efficacemente nell'analisi delle immagini
(visione artificiale).

In breve:
Ispirazione Biologica:
    La sua architettura imita il funzionamento della corteccia visiva
    del cervello umano, che elabora progressivamente le
    informazioni visive da semplici (bordi, linee)
    a complesse (oggetti interi).
    
Filtri e "Convoluzione":
    La caratteristica chiave è l'uso di strati convoluzionali.
    Questi strati applicano dei "filtri" (piccole matrici di numeri)
    scorrendo sull'immagine originale per rilevare caratteristiche
    specifiche come bordi, texture o angoli.
    
Riconoscimento Gerarchico:
    Il processo si ripete su più livelli:
        i primi strati imparano a riconoscere pattern semplici,
        mentre gli strati successivi combinano questi pattern
        per identificare forme, nasi, occhi e, infine,
        oggetti completi (es. un volto, un cane, un'auto).
                          
Applicazioni:
    È la tecnologia dietro il riconoscimento facciale,
    auto a guida autonoma,
    diagnostica medica tramite immagini (radiografie, risonanze magnetiche)
    filtri di Instagram/Snapchat
'''                                                                                                                                                                                                                                                

"""
CNN (Convolutional Neural Network)
CNN di classificazione immagini (cani/gatti/cavalli/uccelli) in PyTorch,
usando una sola cartella 'images_animali/' e split train/val da codice.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# --- 1. Trasformazioni immagini ---
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),              # converte in [0,1]
])

# --- 2. Dataset unico + split train/val ---
# Assicurati che esista la cartella:
# images_animali/cani, images_animali/gatti, etc.
root_dir = "images_animali"

full_ds = datasets.ImageFolder(root=root_dir, transform=transform)
num_classes = len(full_ds.classes)
print("Classi trovate:", full_ds.classes)

# 80% train, 20% val
train_size = int(0.8 * len(full_ds))
val_size = len(full_ds) - train_size

train_ds, val_ds = random_split(full_ds, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=32)

# --- 3. Definizione modello CNN ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),          # 128 -> 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),          # 64 -> 32
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = SimpleCNN(num_classes=num_classes).to(device)

# --- 4. Loss & optimizer ---
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# --- 5. Training loop ---
EPOCHS = 5  # tienere basso per le demo

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_ds)
    train_acc = correct / total

    # Valutazione
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_acc = val_correct / val_total
    print(
        f"Epoch {epoch}/{EPOCHS} "
        f"- loss={train_loss:.4f} "
        f"- acc={train_acc:.3f} "
        f"- val_acc={val_acc:.3f}"
    )

# --- 6. Salvataggio modello ---
torch.save(model.state_dict(), "cnn_animali.pth")
print("Modello salvato in cnn_animali.pth")

