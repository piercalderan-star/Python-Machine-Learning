'''
Un modello Seq2Seq (Sequence-to-Sequence, Sequenza a Sequenza) è un'architettura neurale avanzata utilizzata per compiti in cui l'input è una sequenza e l'output è anch'esso una sequenza diversa, come la traduzione automatica (es. Italiano -> Inglese) o la sintesi di testo.
È composto da due parti principali:
Encoder (Codificatore): Legge la sequenza di input e la comprime in un unico vettore di contesto (context vector).
Decoder (Decodificatore): Prende il vettore di contesto e lo espande, generando la sequenza di output desiderata, parola per parola.
Questo esempio è significativamente più complesso dei precedenti perché coinvolge: tokenizzazione, padding, embedding, RNN (useremo GRU per semplicità), e un ciclo di training che gestisce entrambe le parti del modello.
Esempio Python: Traduzione Semplice (Inglese -> Italiano) con Seq2Seq in PyTorch
Useremo un dataset molto piccolo e fittizio per dimostrare il concetto.
'''

import torch
import torch.nn as nn
import torch.optim as optim
import random
import spacy # Useremo spacy per la tokenizzazione reale (opzionale, ma consigliato)

# --- 1. Preparazione Dati e Tokenizzazione ---

# Coppie di frasi di esempio (Input Inglese, Output Italiano)
raw_data = [
    ("hi, how are you?", "ciao, come stai?"),
    ("i am fine.", "sto bene."),
    ("what is your name?", "come ti chiami?"),
    ("my name is bob.", "mi chiamo bob."),
    ("where do you live?", "dove vivi?"),
    ("i live in rome.", "vivo a roma."),
    ("good morning", "buongiorno"),
    ("good night", "buonanotte")
]

# Definizione di token speciali
PAD_token = 0  # Usato per riempire le frasi alla stessa lunghezza
SOS_token = 1  # Start Of Sentence token
EOS_token = 2  # End Of Sentence token

class Language:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS"}
        self.n_words = 3  # Contiamo già PAD, SOS, EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def prepareData(data):
    input_lang = Language('eng')
    output_lang = Language('ita')
    
    pairs = []
    for eng, ita in data:
        input_lang.addSentence(eng)
        output_lang.addSentence(ita)
        pairs.append((eng, ita))
    
    return input_lang, output_lang, pairs

input_lang, output_lang, pairs = prepareData(raw_data)
print(f"Vocabolario Inglese (Input): {input_lang.n_words} parole")
print(f"Vocabolario Italiano (Output): {output_lang.n_words} parole")

# --- Funzioni di conversione testo -> tensore ---

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token) # Aggiungi token di fine frase
    result = torch.tensor(indexes, dtype=torch.long).view(-1, 1)
    return result

def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)


# --- 2. Definizione del Modello Seq2Seq (Encoder e Decoder GRU) ---

class EncoderGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderGRU, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)

class DecoderGRU(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderGRU, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = torch.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)

# --- 3. Ciclo di Addestramento ---

HIDDEN_SIZE = 128
encoder = EncoderGRU(input_lang.n_words, HIDDEN_SIZE)
decoder = DecoderGRU(HIDDEN_SIZE, output_lang.n_words)
optimizer = optim.SGD(list(encoder.parameters()) + list(decoder.parameters()), lr=0.01)
criterion = nn.NLLLoss() # Negative Log Likelihood Loss (standard per classificazione/softmax)

NUM_ITERS = 5000 # Numero di iterazioni di addestramento
print(f"\nInizio addestramento per {NUM_ITERS} iterazioni...")

def train(input_tensor, target_tensor, encoder, decoder, optimizer, criterion, max_length=10):
    encoder_hidden = encoder.initHidden()
    optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    loss = 0

    # Encoding: elabora la frase di input parola per parola
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)

    # Decoding: genera la frase di output parola per parola
    decoder_input = torch.tensor([[SOS_token]]) # Inizia con SOS token
    decoder_hidden = encoder_hidden # Lo stato finale dell'encoder diventa lo stato iniziale del decoder

    for di in range(target_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()  # usa la previsione come input successivo

        loss += criterion(decoder_output, target_tensor[di])
        if decoder_input.item() == EOS_token:
            break

    loss.backward()
    optimizer.step()

    return loss.item() / target_length

# Esegui il training loop
for iter in range(1, NUM_ITERS + 1):
    training_pair = tensorsFromPair(random.choice(pairs))
    input_tensor = training_pair[0]
    target_tensor = training_pair[1]

    loss = train(input_tensor, target_tensor, encoder, decoder, optimizer, criterion)
    
    if iter % 1000 == 0:
        print(f"Iterazione {iter}, Loss media: {loss:.4f}")

print("Addestramento completato.")

# --- 4. Funzione di Valutazione/Traduzione (Inference) ---

def evaluate(encoder, decoder, sentence, max_length=10):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)

        decoder_input = torch.tensor([[SOS_token]])
        decoder_hidden = encoder_hidden

        decoded_words = []

        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return ' '.join(decoded_words)

# --- 5. Test Finale ---
testo=input("scrivi in inglese: ")
evaluate(encoder, decoder, testo)


print("\n--- Test Traduzione ---")
print(f"Input: 'i live in rome.' -> Output: {evaluate(encoder, decoder, 'i live in rome.')}")
print(f"Input: 'what is your name?' -> Output: {evaluate(encoder, decoder, 'what is your name?')}")
print(f"Input: 'i am fine.' -> Output: {evaluate(encoder, decoder, 'i am fine.')}")

while True:
    user_input = input("\nInserisci frase (ENG): ").strip().lower()
    
    if user_input in ['esci', 'exit', 'quit']:
        print("Uscita...")
        sys.exit()

    if not user_input:
        continue
        
    # Esegui la traduzione
    translation = evaluate(encoder, decoder, user_input)
    print(f"Traduzione (ITA): {translation}")




