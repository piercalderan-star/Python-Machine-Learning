# genai_api.py
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

client = OpenAI()
app = FastAPI(title="Mini Chat API GPT")

class Query(BaseModel):
    message: str

@app.post("/chat")
def chat(q: Query):
    res = client.chat.completions.create(
        model="gpt-4o-mini",  # o altro modello abilitato
        messages=[
            {"role": "system", "content": "Sei un assistente gentile e conciso."},
            {"role": "user", "content": q.message},
        ],
        temperature=0.7,
        max_tokens=200,
    )

    # 👇 Qui la sintassi corretta con la nuova libreria
    answer = res.choices[0].message.content

    return {"response": answer}


