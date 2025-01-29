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

(deformed, distorted, disfigured:1.3), (poor proportions:1.2), 
(unnatural anatomy:1.3), (asymmetrical features:1.1), (unrealistic hands:1.3), 
(awkward posture:1.2), (unnatural facial proportions:1.3), (elongated neck:1.2), 
(slouched shoulders:1.1), (crooked tie:1.1), (improper suit fit:1.2), (low resolution, blurry, pixelated:1.3), 
(cartoonish, painting, drawing, anime:1.4), (harsh lighting, overexposed:1.1), (busy background:1.2), 
(distorted perspective:1.1), (unprofessional appearance:1.2), (makeup, styled hair:1.1), (logo, pattern, accessory:1.2), 
(dramatic expression:1.2), (tilted head:1.1), (body rotation:1.1), (bowtie, patterned tie:1.3), (improper tie width:1.1), 
(3D rendering artifacts:1.1), (Unnatural limb lengths:1.3), (Compressed torso:1.2), (Irregular shoulder width:1.2), 
(Forced hand positions:1.3), (Unrealistic shadow casting:1.1), (Unnatural jawline:1.1), (Facial distortion:1.2), (Inconsistent scale:1.3), 
(Unbalanced features:1.2), (Abnormal eye spacing:1.1), (Unnatural cheekbones:1.1), (Stretched silhouette:1.2)


"""
