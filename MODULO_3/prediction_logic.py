import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Definisci le classi in base al tuo addestramento (nel nostro caso 3 classi)
CLASSES = ['Cane', 'Gatto', 'Uccello', 'Cavallo']

# Definisci la stessa architettura di rete usata in fase di addestramento
class AnimalClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(AnimalClassifier, self).__init__()
        # Una CNN molto semplice per l'esempio
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 32 * 32, num_classes) # Adatta in base alla dimensione immagine/pooling

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 16 * 32 * 32) # Flatten
        x = self.fc1(x)
        return x

# Funzione per caricare il modello addestrato (SIMULATO/Sostituito)
def load_model():
    model = AnimalClassifier(num_classes=len(CLASSES))
    # NOTA: Qui caricheresti il tuo modello reale, ad esempio:
    # model.load_state_dict(torch.load('modello_animali.pth', map_location=torch.device('cpu')))
    # Per l'esempio, usiamo pesi casuali
    return model

# Funzione per preprocessare l'immagine e fare la previsione
def predict_image(image_path, model):
    # Definire le stesse trasformazioni usate durante l'addestramento
    transform = transforms.Compose([
        transforms.Resize((64, 64)), # Assicurati che sia la dimensione corretta
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_t = transform(image)
    image_t = image_t.unsqueeze(0) # Aggiunge una dimensione batch (1 immagine)

    # Previsione
    model.eval()
    with torch.no_grad():
        outputs = model(image_t)
        # Calcola le probabilità usando Softmax
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
        
        # Trova la classe con la probabilità più alta
        _, predicted_idx = torch.max(outputs, 1)
        predicted_class = CLASSES[predicted_idx.item()]
        confidence = probabilities[predicted_idx.item()]

    return predicted_class, confidence, probabilities
