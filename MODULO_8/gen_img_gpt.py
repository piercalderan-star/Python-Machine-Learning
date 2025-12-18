#pip install diffusers accelerate torch

from openai import OpenAI
client = OpenAI()

prompt = "Un robot che studia machine learning davanti a un computer, stile cyberpunk."

res = client.images.generate(
    model="gpt-image-1",
    prompt=prompt,
    size="1024x1024"
)

print(res.data[0].url)
