from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32
)

pipe = pipe.to("cpu")  # o "cuda" se hai GPU

prompt = "una città futuristica vista al tramonto"
image = pipe(prompt).images[0]
image.save("output.png")
