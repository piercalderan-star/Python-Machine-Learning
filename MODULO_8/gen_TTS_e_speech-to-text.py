from openai import OpenAI

client = OpenAI()

text = "Questo è un test di sintesi vocale."

with client.audio.speech.with_streaming_response.create(
    model="gpt-4o-mini-tts",
    voice="alloy",
    input=text,
    response_format="mp3",
) as resp:
    resp.stream_to_file("audio_tts.mp3")

print("File audio_tts.mp3 salvato correttamente (streaming).")



from openai import OpenAI
client = OpenAI()

audio_file = open("audio_tts.mp3", "rb")

res = client.audio.transcriptions.create(
    model="gpt-4o-mini-transcribe",
    file=audio_file
)

print(res.text)
