import pandas as pd
from sentence_transformers import SentenceTransformer, util

df = pd.read_csv("faq_prodotti.csv")   # domanda, risposta

model = SentenceTransformer("hf/all-MiniLM-L6-v2")
faq_embed = model.encode(df["domanda"].tolist(), convert_to_tensor=True)

query = "Quanto dura la ricarica?"
query_emb = model.encode(query, convert_to_tensor=True)

scores = util.cos_sim(query_emb, faq_embed)[0]
top_idx = int(scores.argmax())

print("Domanda più simile:", df.loc[top_idx, "domanda"])
print("Risposta:", df.loc[top_idx, "risposta"])
