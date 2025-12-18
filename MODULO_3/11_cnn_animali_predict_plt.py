# cnn_animali_predict.py
"""
Carica il modello CNN allenato (cnn_animali.pth) e
fa la previsione sulle immagini presenti nella cartella 'predict_images'.

Struttura attesa:
- images_animali/  -> usata per recuperare l'ordine delle classi
- predict_images/  -> contiene le immagini da classificare
"""

import os
import glob
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from PIL import Image
import matplotlib.pyplot as plt

# --- Configurazione di base ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "cnn_animali.pth"
DATASET_ROOT = "images_animali"
PREDICT_DIR = "predict_images"

print("Device:", DEVICE)

# --- Trasformazione immagine (deve essere la stessa del training) ---
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# --- Recupero classi dal dataset di training ---
if not os.path.isdir(DATASET_ROOT):
    raise FileNotFoundError(
        f"Cartella '{DATASET_ROOT}' non trovata. "
        "Assicurarsi che esista e contenga le sottocartelle delle classi."
    )

ds = datasets.ImageFolder(root=DATASET_ROOT, transform=transform)
class_names = ds.classes
num_classes = len(class_names)
print("Classi trovate:", class_names)


# --- Definizione modello (deve essere identico a quello usato in training) ---
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


# --- Caricamento modello ---
model = SimpleCNN(num_classes=num_classes).to(DEVICE)
if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"File modello '{MODEL_PATH}' non trovato. "
        "Allenare prima con cnn_animali_pytorch.py."
    )

state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval()

print(f"Modello caricato da {MODEL_PATH}")

#print(state_dict) #oltre 700 righe!

# --- Funzione di predizione per una singola immagine ---
def predict_image(path: str):
    img = Image.open(path).convert("RGB")
    img_t = transform(img).unsqueeze(0).to(DEVICE)  # shape: (1,3,128,128)

    with torch.no_grad():
        logits = model(img_t)
        probs = torch.softmax(logits, dim=1)[0]  # vettore (num_classes,)

    # indice della classe con probabilità massima
    top_idx = torch.argmax(probs).item()
    top_class = class_names[top_idx]
    top_prob = probs[top_idx].item()

    return top_class, top_prob, probs


# --- Loop sulle immagini in PREDICT_DIR ---
if not os.path.isdir(PREDICT_DIR):
    raise FileNotFoundError(
        f"Cartella '{PREDICT_DIR}' non trovata. Creala e aggiungi alcune immagini."
    )

image_paths = []
for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
    image_paths.extend(glob.glob(os.path.join(PREDICT_DIR, ext)))

if not image_paths:
    raise RuntimeError(
        f"Nessuna immagine trovata in '{PREDICT_DIR}'. "
        "Aggiungi qualche file .jpg/.png e riprova."
    )

print(f"Trovate {len(image_paths)} immagini in '{PREDICT_DIR}'\n")


animal_perc=[]
animal_name=[]
for path in image_paths:
    top_class, top_prob, _ = predict_image(path)
    print(f"Immagine: {os.path.basename(path)}")
    print(f" → Predizione: {top_class} ({top_prob*100:.1f}%)")
    print("-" * 40)
    animal_name.append(top_class)
    animal_perc.append(f"{top_prob*100:.1f}")

print(animal_name)
print(animal_perc)

plt.bar(animal_name,animal_perc)
plt.xlabel("animali")
plt.ylabel("%")
plt.show()
