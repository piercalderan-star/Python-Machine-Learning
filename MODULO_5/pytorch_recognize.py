import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from torchvision.datasets import ImageFolder
from PIL import Image

# 1. Caricamento del modello locale da Hugging Face
# Sostituisci con il percorso della cartella dove hai clonato il modello
model_path = "microsoft/resnet-18" 

# Carichiamo il processore per la pre-elaborazione corretta (resize, normalizzazione)
processor = AutoImageProcessor.from_pretrained(model_path)

# 2. Identificazione delle classi dal tuo dataset locale
dataset_dir = "dataset_pc"
temp_ds = ImageFolder(root=dataset_dir)

labels = temp_ds.classes

id2label = {i: label for i, label in enumerate(labels)}

label2id = {label: i for i, label in enumerate(labels)}

print(labels)

# 3. Caricamento del modello con la testa di classificazione corretta
model = AutoModelForImageClassification.from_pretrained(
    model_path,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True # Necessario se il numero di classi è diverso dall'originale
)
model.eval()

# 4. Funzione di predizione
def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    
    # Pre-elaborazione (ritorno in formato PyTorch 'pt')
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
    predicted_class_idx = logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]

# Esempio d'uso
#print(f"Risultato: {predict('mouse.jpg')}")
print(f"Risultato: {predict('tastiera.jpg')}")





















