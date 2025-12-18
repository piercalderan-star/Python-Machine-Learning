##import torch
##from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
##
### Cambia il modello a mT5-base (multilingua)
##model_name = "google/mt5-base"
##
### Carica il tokenizer e il modello
##tokenizer = AutoTokenizer.from_pretrained(model_name)
### Usa device_map="auto" se hai una GPU, altrimenti lascialo vuoto per usare la CPU
##model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto")
##
### --- Esempio 1: Traduzione ---
##input_text = "Traduci in francese: Come stai oggi?"
##input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
### Aggiungi parametri per una generazione più fluida
##outputs = model.generate(input_ids, max_new_tokens=50, num_beams=5, early_stopping=True)
##generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
##
##print(f"Input: {input_text}")
##print(f"Output generato: {generated_text}")
### Output previsto: Comment allez-vous aujourd'hui?
##
##print("-" * 20)
##
### --- Esempio 2: Riassunto ---
##input_text_2 = "Crea un riassunto di questa frase: L'intelligenza artificiale sta rivoluzionando molti settori, dall'assistenza sanitaria ai trasporti, offrendo nuove efficienze e capacità."
##input_ids_2 = tokenizer(input_text_2, return_tensors="pt").input_ids.to(model.device)
##outputs_2 = model.generate(input_ids_2, max_new_tokens=50, num_beams=5, early_stopping=True)
##generated_text_2 = tokenizer.decode(outputs[0], skip_special_tokens=True) # Usa outputs[0] corretto
##
##print(f"Input 2: {input_text_2}")
##print(f"Output generato 2: {generated_text_2}")
### Output previsto: L'intelligenza artificiale sta rivoluzionando molti settori, offrendo nuove efficienze e capacità.

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 1. Specifica il nome del modello
model_name = "google/flan-t5-base"
#model_name = "mt5-base"

# 2. Carica il tokenizer e il modello pre-addestrato
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Usa device_map="auto" per caricare il modello su GPU/CPU automaticamente
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto")

# 3. Definisci il testo di input (prompt)
# Flan-T5 è ottimizzato per istruzioni (instruction-tuned)
# quindi è utile formulare il prompt come un'istruzione
input_text = "Traduci in francese: Come stai oggi?"

# 4. Tokenizza l'input
# return_tensors="pt" restituisce tensori PyTorch
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)

# 5. Genera l'output
# max_new_tokens controlla la lunghezza massima del testo generato
outputs = model.generate(input_ids, max_new_tokens=100)

# 6. Decodifica l'output generato in testo leggibile
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# 7. Stampa il risultato
print(f"Input: {input_text}")
print(f"Output generato: {generated_text}")

# Esempio aggiuntivo con un'altra istruzione
input_text_2 = "Crea un riassunto di questa frase: L'intelligenza artificiale sta rivoluzionando molti settori, dall'assistenza sanitaria ai trasporti, offrendo nuove efficienze e capacità."
input_ids_2 = tokenizer(input_text_2, return_tensors="pt").input_ids.to(model.device)
outputs_2 = model.generate(input_ids_2, max_new_tokens=50)
generated_text_2 = tokenizer.decode(outputs_2[0], skip_special_tokens=True)

print(f"\nInput 2: {input_text_2}")
print(f"Output generato 2: {generated_text_2}")

##from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
##
##model_name = "google/flan-t5-base"
##
##tokenizer = AutoTokenizer.from_pretrained(model_name)
##model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
##
##prompt = "Spiega in modo semplice cos'è il machine learning."
##
##inputs = tokenizer(prompt, return_tensors="pt")
##outputs = model.generate(**inputs, max_new_tokens=150)
##
##print(tokenizer.decode(outputs[0], skip_special_tokens=True))
