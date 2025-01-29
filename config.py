import os
import torch

# Define paths for models
MODEL_PATH = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
LORA_PATH = "models/lora/sd_xl_offset_example-lora_1.0.safetensors"
YOLO_PATH = "yolov8n-face-lindevs.pt"
DEBUG_IMAGE = "debug_images"

# Define URLs for model downloads
LORA_DOWNLOAD_URL = "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_offset_example-lora_1.0.safetensors"

# Configure device preferences here, I'm using MPS for Apple Silicon GPU:
ALLOW_CUDA = True
ALLOW_MPS = False

if torch.cuda.is_available() and ALLOW_CUDA:
    DEVICE = "cuda"
elif torch.backends.mps.is_available() and ALLOW_MPS:
    DEVICE = "mps"
else:
    DEVICE = "cpu"


PROMPT = """

A photorealistic portrait of a 23-year-old male model from Canada in a tailored navy suit with a matching tie
He stands confidently against a light gray studio backdrop, illuminated by soft, diffused lighting that highlights the suit's texture and fit. 
The image is captured in a medium shot, 8K HDR, showcasing the refined elegance of his attire.

"""


PROMPT1 = """

A photorealistic portrait of a 23-year-old male model from Brazil, dressed in a tailored charcoal suit and matching tie that conveys sophistication.
He is positioned against a minimalist black background in a well-lit studio, with dramatic lighting that highlights the contours of his suit and creates a striking visual contrast. 
The image is captured in a three-quarter angle, 8K HDR, emphasizing the suit's craftsmanship.

"""

PROMPT2 = """

A photorealistic portrait of a 23-year-old male model from Germany, presenting a polished look in a tailored light gray suit and matching tie.
He stands against a soft white backdrop in a professional studio setting, enhanced by bright, focused lighting that accentuates the suit's details and creates a clean, modern aesthetic.
The image is captured in a close-up shot, 8K HDR, reflecting the suit's elegance.

"""

PROMPT3 = """

A photorealistic portrait of a 23-year-old male model from Japan, showcasing a tailored black suit and matching tie that radiates confidence. 
He is set against a deep blue background in a contemporary photo studio, with strategic lighting that highlights the suit's sharp lines and luxurious fabric. 
The image is captured in a full-body shot, 8K HDR, illustrating the suit's sophisticated design.

"""

PROMPT4 = """




"""



NPROMPT = """

(deformed iris, deformed pupils, deformed hands:1.3), 
(asymmetrical eyes, unnatural facial proportions:1.2), (poor posture, slouching shoulders:1.1), 
(improper suit tailoring, wrinkled fabric:1.2), (low resolution, blurry, pixelated:1.3),
(cartoonish rendering, 3D model, drawing:1.4), (harsh lighting, overexposed:1.1), 
(busy background, cluttered environment:1.2), (unprofessional appearance, casual clothing:1.2), 
(makeup, styled hair, accessories:1.1), (bowtie, patterned tie, wide tie:1.3), (unnatural jawline, facial distortion:1.2), 
(stretched limbs, compressed torso:1.3), (awkward hand position, missing fingers:1.4), (body tilt, angled posture:1.1), 
(shadow artifacts, rendering glitches:1.1)

"""



SEED = 42
DENOISING_STRENGTH = 0.8
LORA_ALPHA = 0.85
LORA_SCALE = 0.75