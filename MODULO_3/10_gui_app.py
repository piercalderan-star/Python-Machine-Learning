import tkinter as tk
from tkinter import filedialog, Label, messagebox
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np

# --- Funzioni di supporto (Devono essere le stesse dell'addestramento) ---
CLASSES = []
try:
    with open('animal_classes.txt', 'r') as f:
        CLASSES = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    messagebox.showerror("Errore", "Impossibile trovare animal_classes.txt. Esegui prima train_model.py")
    exit()

class AnimalClassifier(nn.Module):
    def __init__(self, num_classes):
        super(AnimalClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 32 * 32, num_classes) 

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 16 * 32 * 32)
        x = self.fc1(x)
        return x

def load_model(model_path='animal_classifier.pth'):
    model = AnimalClassifier(num_classes=len(CLASSES))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval() # Imposta in modalità valutazione
    return model

def predict_image(image_path, model):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_t = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_t)
        probabilities = torch.nn.functional.softmax(outputs, dim=1).flatten() * 100
        _, predicted_idx = torch.max(outputs, 1)
        predicted_class = CLASSES[predicted_idx.item()]
        confidence = probabilities[predicted_idx.item()]

    return predicted_class, confidence, probabilities

# --- Logica GUI Tkinter ---
model = load_model()

def open_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if not file_path:
        return

    predicted_class, confidence, all_probabilities = predict_image(file_path, model)

    display_image(file_path)
    
    details_text = f"Risultato:\n"
    for i, class_name in enumerate(CLASSES):
        details_text += f"- {class_name}: {all_probabilities[i]:.2f}%\n"

    result_label.config(text=details_text)

def display_image(file_path):
    img = Image.open(file_path)
    img = img.resize((300, 300), Image.Resampling.LANCZOS)
    photo = ImageTk.PhotoImage(img)
    image_label.config(image=photo)
    image_label.image = photo 

# Configurazione della finestra principale Tkinter
root = tk.Tk()
root.title("Classificatore Animali PyTorch GUI")
root.geometry("400x550") 

open_button = tk.Button(root, text="Apri Immagine Animale", command=open_image)
open_button.pack(pady=10)

image_label = Label(root, bg='gray', width=300, height=300)
image_label.pack(pady=10)

result_label = Label(root, text="Seleziona un'immagine per l'analisi.", font=("Helvetica", 12), justify=tk.LEFT)
result_label.pack(pady=10)

root.mainloop()


##import tkinter as tk
##from tkinter import filedialog, Label, messagebox
##from PIL import Image, ImageTk
##import prediction_logic as pl # Importa il file di logica creato sopra
##
### Carica il modello una volta all'avvio dell'app
##try:
##    model = pl.load_model()
##    print("Modello caricato con successo.")
##except Exception as e:
##    messagebox.showerror("Errore Caricamento Modello", f"Impossibile caricare il modello: {e}")
##    exit()
##
##def open_image():
##    # Apre una finestra di dialogo per selezionare un file
##    file_path = filedialog.askopenfilename(
##        filetypes=[("Image files", "*.jpg *.png *.jpeg")]
##    )
##    if not file_path:
##        return
##
##    # Esegue la predizione
##    predicted_class, confidence, all_probabilities = pl.predict_image(file_path, model)
##
##    # Aggiorna l'interfaccia grafica
##    display_image(file_path)
##    result_text = f"Previsione: {predicted_class}\nCertezza: {confidence:.2f}%"
##    
##    # Aggiungi tutte le percentuali
##    details_text = "\n\nPercentuali Dettagliate:\n"
##    for i, class_name in enumerate(pl.CLASSES):
##        details_text += f"- {class_name}: {all_probabilities[i]:.2f}%\n"
##
##    result_label.config(text=result_text + details_text)
##
##def display_image(file_path):
##    # Carica l'immagine originale e la ridimensiona per la GUI
##    img = Image.open(file_path)
##    img = img.resize((300, 300), Image.Resampling.LANCZOS)
##    photo = ImageTk.PhotoImage(img)
##
##    # Aggiorna la label dell'immagine nella GUI
##    image_label.config(image=photo)
##    image_label.image = photo # Mantiene un riferimento per evitare garbage collection
##
### Configurazione della finestra principale Tkinter
##root = tk.Tk()
##root.title("Classificatore Animali PyTorch GUI")
##root.geometry("400x550") # Larghezza x Altezza
##
### Pulsante per aprire l'immagine
##open_button = tk.Button(root, text="Apri Immagine Animale", command=open_image)
##open_button.pack(pady=10)
##
### Etichetta per visualizzare l'immagine caricata
##image_label = Label(root, bg='gray', width=300, height=300)
##image_label.pack(pady=10)
##
### Etichetta per mostrare i risultati della previsione
##result_label = Label(root, text="Seleziona un'immagine per l'analisi.", font=("Helvetica", 12))
##result_label.pack(pady=10)
##
### Avvia il loop principale della GUI
##root.mainloop()
