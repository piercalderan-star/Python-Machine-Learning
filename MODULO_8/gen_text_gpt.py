# gen_text_gpt.py
from openai import OpenAI

# Usa la chiave letta dalla variabile d'ambiente OPENAI_API_KEY
client = OpenAI()

prompt = "Scrivi un breve testo motivazionale di 3 frasi sullo studio del Machine Learning."

res = client.chat.completions.create(
    model="gpt-4o-mini",        # o altro modello abilitato nel tuo account
    messages=[
        {"role": "system", "content": "Sei un assistente gentile e motivante."},
        {"role": "user", "content": prompt},
    ],
    temperature=0.7,
    max_tokens=150,
)

# Sintassi corretta con la nuova libreria
print(res.choices[0].message.content)
