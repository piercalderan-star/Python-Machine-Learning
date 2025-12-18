'''
La similarità sentence-transformer
 è un metodo per misurare quanto due frasi
 siano semanticamente (nel significato) simili,
 utilizzando modelli di deep learning chiamati "sentence transformers". 

Il processo si svolge in due fasi principali:
 Generazione di embedding (vettori):
    Ogni frase viene convertita in un vettore numerico a lunghezza fissa,
    chiamato embedding.
    Questo vettore rappresenta il significato
    semantico dell'intera frase in uno spazio
    vettoriale multidimensionale.

 Calcolo della similarità:
    La similarità tra due frasi viene calcolata misurando
    la distanza o l'angolo tra i loro rispettivi vettori
    di embedding in questo spazio.
    La misura più comune è la similarità
    del coseno (cosine similarity).
    Un punteggio elevato (vicino a 1) indica alta similarità,
    mentre un punteggio basso (vicino a 0 o negativo) indica bassa similarità
    o significati opposti. 

 Vantaggi
I modelli sentence-transformers sono specificamente ottimizzati
per questo compito e offrono diversi vantaggi rispetto
ai tradizionali modelli transformer (come il BERT vanilla):
    
Efficienza
Sono molto più veloci nel generare embedding
per molte frasi, poiché non richiedono un confronto incrociato (cross-encoding)
di tutte le coppie possibili.

Prestazioni
    Sono addestrati per produrre embedding i cui confronti diretti
    (tramite similarità del coseno)
    riflettono accuratamente la similarità semantica umana.
'''

##Esempio 1
##from sentence_transformers import SentenceTransformer, util
##
##model = SentenceTransformer("hf/all-MiniLM-L6-v2")
##
##s1 = "Il telefono è bagnato."
##s2 = "Il mio smartphone è caduto nell'acqua."
##
##s3 = "Il mio cellulare è bagnato."
##s4 = "Il mio cellulare è nuovo."
##
##e1 = model.encode(s1)
##e2 = model.encode(s2)
##
##e3 = model.encode(s3)
##e4 = model.encode(s4)
##
##
##sim = util.cos_sim(e1, e2)
##print("Similarità:", float(sim))
##
##sim2 = util.cos_sim(e3, e4)
##print("Similarità:", float(sim2))












##Esempio 2
##from sentence_transformers import SentenceTransformer, util
##
### 1. Carica un modello pre-addestrato
### 'all-MiniLM-L6-v2' è un modello leggero ed efficiente, ottimo per iniziare.
##print("Caricamento del modello Sentence Transformer...")
##model = SentenceTransformer('hf/all-MiniLM-L6-v2')
##print("Modello caricato con successo.\n")
##
### 2. Definisci le due coppie di frasi da confrontare
##
### Coppia A: Molto simili semanticamente
##sentence_a1 = "Il gatto si siede sul tappeto."
##sentence_a2 = "Un felino domestico riposa su una stuoia."
##
### Coppia B: Diverse semanticamente
##sentence_b1 = "La programmazione in Python è divertente."
##sentence_b2 = "Il meteo per domani prevede pioggia."
##
##print(f"Coppia A (Simili):\n  - '{sentence_a1}'\n  - '{sentence_a2}'")
##print(f"Coppia B (Diverse):\n  - '{sentence_b1}'\n  - '{sentence_b2}'\n")
##
### 3. Codifica le frasi in embedding (vettori)
### Creiamo una lista unica di tutte le frasi per efficienza
##sentences = [sentence_a1, sentence_a2, sentence_b1, sentence_b2]
##embeddings = model.encode(sentences, convert_to_tensor=True)
##
### Estrai gli embedding specifici per le coppie
##emb_a1 = embeddings[0]
##emb_a2 = embeddings[1]
##emb_b1 = embeddings[2]
##emb_b2 = embeddings[3]
##
### 4. Calcola la similarità del coseno
### util.cos_sim calcola la similarità tra due tensori
### Il risultato è un tensore 2D, estraiamo il valore float con .item()
##similarity_a = util.cos_sim(emb_a1, emb_a2).item()
##similarity_b = util.cos_sim(emb_b1, emb_b2).item()
##
### 5. Stampa i risultati float
##print("-" * 40)
##print(f"Punteggio Similarità Coppia A: {similarity_a:.4f}")
##print(f"Punteggio Similarità Coppia B: {similarity_b:.4f}")
##print("-" * 40 + "\n")
##
### 6. Spiegazione dei risultati (L'interpretazione)
##
##print("### Spiegazione dei Risultati ###")
##
### --- Spiegazione Coppia A ---
##print(f"\nCoppia A (Punteggio: {similarity_a:.4f}):")
##print("Le frasi 'Il gatto si siede sul tappeto' e 'Un felino domestico riposa su una stuoia'")
##print("hanno ottenuto un punteggio molto alto (vicino a 1.0).")
##print("Questo accade perché, nonostante utilizzino parole diverse (*gatto/felino*, *siede/riposa*, *tappeto/stuoia*),")
##print("i modelli *Sentence Transformers* sono stati addestrati per catturare il **significato semantico**.")
##print("Il modello ha riconosciuto che entrambe le frasi descrivono la stessa scena o concetto, posizionando i loro vettori molto vicini nello spazio semantico.")
##
### --- Spiegazione Coppia B ---
##print(f"\nCoppia B (Punteggio: {similarity_b:.4f}):")
##print("Le frasi 'La programmazione in Python è divertente' e 'Il meteo per domani prevede pioggia'")
##print("hanno ottenuto un punteggio basso (vicino a 0.0, o talvolta negativo, a seconda del modello).")
##print("Questo indica una bassa similarità semantica.")
##print("Gli argomenti (programmazione vs. meteo) sono completamente diversi e non correlati. I loro vettori sono quasi ortogonali (ad angolo retto) nello spazio semantico.")




##Esempio 3
##import torch
##import matplotlib.pyplot as plt
##import numpy as np
##from sentence_transformers import SentenceTransformer, util
##from sklearn.decomposition import PCA
##
#### 1. Carica il modello
##model = SentenceTransformer('hf/all-MiniLM-L6-v2')
##
#### 2. Definisci le frasi
##sentences = [
##    "Il gatto si siede sul tappeto.",  # A1
##    "Un felino domestico riposa su una stuoia.", # A2 (Simile a A1)
##    "La programmazione in Python è divertente.", # B1
##    "Il meteo per domani prevede pioggia." # B2 (Diversa da tutti)
##]
##
#### 3. Calcola gli embedding (vettori 384-dimensionali)
##embeddings = model.encode(sentences, convert_to_tensor=True)
##
#### 4. Riduzione della dimensionalità con PCA a 2D
#### Questo passo è fondamentale per poter visualizzare i vettori su un grafico 2D
##pca = PCA(n_components=2)
##embeddings_2d = pca.fit_transform(embeddings.cpu().numpy())
##
#### Estrai i punti 2D per ogni frase
##p_a1, p_a2, p_b1, p_b2 = embeddings_2d
##
#### 5. Calcola le similarità (per riferimento nell'interpretazione)
##sim_a = util.cos_sim(embeddings[0], embeddings[1]).item()
##sim_b = util.cos_sim(embeddings[2], embeddings[3]).item()
##
#### 6. Disegna il grafico con Matplotlib
##plt.figure(figsize=(10, 8))
##
#### Disegna i vettori come frecce (arrows) partendo dall'origine (0,0)
##plt.arrow(0, 0, p_a1[0], p_a1[1], head_width=0.05, head_length=0.1, color='blue', label=f'A1: "{sentences[0]}"')
##plt.arrow(0, 0, p_a2[0], p_a2[1], head_width=0.05, head_length=0.1, color='cyan', label=f'A2: "{sentences[1]}"')
##plt.arrow(0, 0, p_b1[0], p_b1[1], head_width=0.05, head_length=0.1, color='red', label=f'B1: "{sentences[2]}"')
##plt.arrow(0, 0, p_b2[0], p_b2[1], head_width=0.05, head_length=0.1, color='orange', label=f'B2: "{sentences[3]}"')
##
#### Aggiunge etichette e titolo
##plt.title('Visualizzazione della Similarità del Coseno (Ridotta con PCA)')
##plt.xlabel('Dimensione PCA 1')
##plt.ylabel('Dimensione PCA 2')
##plt.legend(loc='upper right')
##plt.grid(True)
##plt.xlim(-1.5, 1.5)
##plt.ylim(-1.5, 1.5)
##plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
##plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
##plt.axis('equal') # Mantiene gli assi proporzionati per un angolo corretto
##
##plt.show()
##
##print(f"\n--- Punteggi di Similarità Reali (Prima della PCA) ---")
##print(f"Coppia A (Blu/Ciano): {sim_a:.4f} (Alta similarità)")
##print(f"Coppia B (Rosso/Arancio): {sim_b:.4f} (Bassa similarità)")
