from auth_token import auth_token
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from io import BytesIO
import base64 
import os

os.USE_SAFE_TENSORS = False

app = FastAPI()

app.add_middleware(
    CORSMiddleware, 
    allow_credentials=True, 
    allow_origins=["*"], 
    allow_methods=["*"], 
    allow_headers=["*"]
)
# Create a GPU device if it is available
device = "cuda"
model_id = "NiamaLynn/moroccan-interior-painting"
pipe = StableDiffusionPipeline.from_pretrained(model_id, safety_checker=None, use_auth_token=auth_token, use_safetensors=False)
pipe.to(device)

@app.get("/generate")
def generate():
    prompt = "photo of a moroccan riad" 
    with autocast(device): 
        image = pipe(prompt, guidance_scale=10).images[0]

    image.save("testimage.png")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    imgstr = base64.b64encode(buffer.getvalue())

    return Response(content=base64.b64decode(imgstr), media_type="image/png")