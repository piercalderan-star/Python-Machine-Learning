import torch
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from transformers import AutoImageProcessor, AutoModelForImageClassification
from torchvision.datasets import ImageFolder
import os

# --- 1. CONFIGURAZIONE MODELLO E CLASSI ---
model_path = "microsoft/resnet-18" 
dataset_dir = "dataset_pc" # La tua cartella con le sottocartelle componenti

# Caricamento classi dal dataset locale
if not os.path.exists(dataset_dir):
    print(f"Errore: Cartella '{dataset_dir}' non trovata!")
    labels = ["Nessun Dataset"]
else:
    temp_ds = ImageFolder(root=dataset_dir)
    labels = temp_ds.classes

id2label = {i: label for i, label in enumerate(labels)}
label2id = {label: i for i, label in enumerate(labels)}

# Caricamento Modello e Processore
processor = AutoImageProcessor.from_pretrained(model_path)
model = AutoModelForImageClassification.from_pretrained(
    model_path,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)
model.eval()

# --- 2. INTERFACCIA GRAFICA (GUI) ---
class PCComponentApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Riconoscimento Componenti PC 2025")
        self.root.geometry("500x600")
        self.root.configure(bg="#f0f0f0")

        # Titolo
        self.title_label = tk.Label(root, text="AI PC Scanner", font=("Arial", 18, "bold"), bg="#f0f0f0")
        self.title_label.pack(pady=10)

        # Area Immagine
        self.canvas = tk.Canvas(root, width=300, height=300, bg="white", highlightthickness=1)
        self.canvas.pack(pady=10)

        # Etichetta Risultato
        self.result_label = tk.Label(root, text="Seleziona un'immagine per l'analisi", font=("Arial", 12), bg="#f0f0f0")
        self.result_label.pack(pady=10)

        # Bottone Carica
        self.upload_btn = tk.Button(root, text="Carica Immagine", command=self.upload_and_predict, 
                                   font=("Arial", 11), bg="#2196F3", fg="white", padx=20, pady=10)
        self.upload_btn.pack(pady=20)

    def upload_and_predict(self):
        # Selezione File
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        
        if file_path:
            # 1. Visualizzazione Anteprima
            img = Image.open(file_path)
            img.thumbnail((300, 300)) # Ridimensiona per la GUI
            self.img_tk = ImageTk.PhotoImage(img)
            self.canvas.create_image(150, 150, image=self.img_tk)

            # 2. Analisi AI
            try:
                # Pre-elaborazione per ResNet
                image_raw = Image.open(file_path).convert("RGB")
                inputs = processor(images=image_raw, return_tensors="pt")
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    
                predicted_class_idx = logits.argmax(-1).item()
                prediction = model.config.id2label[predicted_class_idx]
                
                # Calcolo confidenza (opzionale)
                probs = torch.nn.functional.softmax(logits, dim=-1)
                confidence = probs[0][predicted_class_idx].item() * 100

                # 3. Aggiornamento UI
                self.result_label.config(text=f"Rilevato: {prediction.upper()}\n(Confidenza: {confidence:.2f}%)", fg="#4CAF50")
            
            except Exception as e:
                messagebox.showerror("Errore", f"Errore durante l'analisi: {e}")

# --- 3. AVVIO APPLICAZIONE ---
if __name__ == "__main__":
    root = tk.Tk()
    app = PCComponentApp(root)
    root.mainloop()
