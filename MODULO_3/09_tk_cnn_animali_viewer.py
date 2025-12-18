# tk_cnn_animali_viewer.py
"""
Piccola demo Tkinter:
- Carica il modello cnn_animali.pth
- Permette di selezionare un'immagine dal disco
- Mostra l'immagine e la predizione (classe + probabilità)
"""

import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox

# --- Config ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "cnn_animali.pth"
DATASET_ROOT = "images_animali"   # per ottenere l'ordine delle classi

# --- Trasformazione immagine (identica a quella del training) ---
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])


# --- Recupero classi dal dataset di training ---
if not os.path.isdir(DATASET_ROOT):
    raise FileNotFoundError(
        f"Cartella '{DATASET_ROOT}' non trovata. "
        "Serve per leggere i nomi delle classi (sottocartelle)."
    )

ds = datasets.ImageFolder(root=DATASET_ROOT, transform=transform)
class_names = ds.classes
num_classes = len(class_names)


# --- Definizione modello (identico a training) ---
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
        "Allenalo e salvalo prima con cnn_animali_pytorch.py."
    )

state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval()


# --- Funzione di predizione per una singola immagine ---
def predict_image(path: str):
    img = Image.open(path).convert("RGB")
    img_t = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(img_t)
        probs = torch.softmax(logits, dim=1)[0]

    top_idx = torch.argmax(probs).item()
    top_class = class_names[top_idx]
    top_prob = probs[top_idx].item()

    # Ritorno anche l'immagine PIL originale per mostrarla in Tkinter
    return img, top_class, top_prob


# --- GUI Tkinter ---
class CNNViewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CNN Animali - Demo PyTorch")
        self.geometry("600x500")

        # Riferimento alla PhotoImage per non farla garbage-collectare
        self.current_image_tk = None

        # Frame immagine
        self.image_label = tk.Label(self, text="Nessuna immagine", bg="gray", width=60, height=20)
        self.image_label.pack(pady=10)

        # Label predizione
        self.pred_label = tk.Label(self, text="Predizione: -", font=("Arial", 14))
        self.pred_label.pack(pady=10)

        # Pulsante selezione
        self.btn_select = tk.Button(self, text="Seleziona immagine...", command=self.on_select_image)
        self.btn_select.pack(pady=5)

    def on_select_image(self):
        filetypes = [("Immagini", "*.jpg *.jpeg *.png *.bmp"), ("Tutti i file", "*.*")]
        path = filedialog.askopenfilename(title="Seleziona immagine", filetypes=filetypes)

        if not path:
            return  # utente ha annullato

        try:
            img, top_class, top_prob = predict_image(path)
        except Exception as e:
            messagebox.showerror("Errore", f"Impossibile analizzare l'immagine:\n{e}")
            return

        # Ridimensiono l'immagine per la GUI (ad es. lato max 300 px)
        max_size = 300
        img_for_tk = img.copy()
        img_for_tk.thumbnail((max_size, max_size), Image.LANCZOS)

        self.current_image_tk = ImageTk.PhotoImage(img_for_tk)
        self.image_label.config(image=self.current_image_tk, text="")

        self.pred_label.config(
            text=f"Predizione: {top_class} ({top_prob*100:.1f}%)"
        )


if __name__ == "__main__":
    app = CNNViewer()
    app.mainloop()
