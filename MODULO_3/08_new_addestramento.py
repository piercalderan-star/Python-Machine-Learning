import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os

# --- Configurazione Dataset e Addestramento ---
DATA_DIR = 'dataset/'
BATCH_SIZE = 16
NUM_EPOCHS = 5 # Aumenta a 15-20 per risultati migliori, 5 è per l'esempio
LEARNING_RATE = 0.001

# Definizione delle trasformazioni: ridimensiona, converti in tensore, normalizza
transform = transforms.Compose([
    transforms.Resize((64, 64)), # Tutte le immagini devono avere la stessa dimensione
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Caricamento del dataset dalle cartelle locali
# ImageFolder assegna automaticamente le etichette (0=cani, 1=gatti, ecc.)
full_dataset = ImageFolder(root=DATA_DIR, transform=transform)

# Suddivisione in Training e Validation Set (80% train, 20% val)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

# Preparazione dei DataLoaders (per caricare i dati in batch durante l'addestramento)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

CLASSES = full_dataset.classes
print(f"Classi trovate: {CLASSES}")
NUM_CLASSES = len(CLASSES)

# --- Definizione del Modello CNN ---
class AnimalClassifier(nn.Module):
    def __init__(self, num_classes):
        super(AnimalClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Il layer lineare è dimensionato per l'output delle convoluzioni (dipende da 64x64 input)
        self.fc1 = nn.Linear(16 * 32 * 32, num_classes) 

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 16 * 32 * 32) # Flatten
        x = self.fc1(x)
        return x

model = AnimalClassifier(NUM_CLASSES)

# --- Addestramento ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("Inizio addestramento...")
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    # Valutazione
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in val_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    print(f'Epoch [{epoch+1}/{NUM_EPOCHS}] Loss: {running_loss/len(train_loader):.3f} Acc. Validazione: {accuracy:.2f}%')

print('Addestramento terminato.')

# --- Salvataggio del Modello e delle Classi ---
MODEL_PATH = 'animal_classifier.pth'
torch.save(model.state_dict(), MODEL_PATH)
# Salviamo anche le classi in un file separato, sono essenziali!
with open('animal_classes.txt', 'w') as f:
    for item in CLASSES:
        f.write("%s\n" % item)

print(f"Modello salvato in {MODEL_PATH}")
print(f"Classi salvate in animal_classes.txt")
