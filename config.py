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

A photorealistic portrait of a 23-year-old male model from Canada in a tailored navy suit with a neatly knotted necktie
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

(worst quality, low quality, normal quality:1.4), (lowres, blurry, unfocused:1.3), 
(unrealistic, cartoon, anime, painting, drawing, sketch, digital art:1.4), 
(deformed iris, deformed pupils, semi-realistic, 3d render:1.3), 
(deformed, distorted, disfigured:1.4), (bad anatomy, extra limbs, mutated hands, hands:1.4), 
text, watermark, logo, signature, jpeg artifacts, chromatic aberration, 
(out of frame:1.2), (poorly drawn face, asymmetrical features:1.3), 
(grain, noise, film grain, motion blur:1.2), (anatomical errors, proportion issues:1.4), 
(asymmetrical shoulders, stretched torso, compressed limbs:1.3), 
(casual clothing, wrinkled fabric, unprofessional attire:1.4), 
(messy hair, heavy makeup, sunglasses, hat, bow tie:1.4), 
(cluttered background, outdoor setting, complex environment:1.4), 
(harsh lighting, overexposed, lens flare, uneven shadows:1.3), 
(side profile, extreme angles, tilted posture:1.4), 
(crossed arms, casual stance, exaggerated expressions:1.3), 
(over-edited, oversaturated, artificial contrast:1.2), 
easynegative, bad-image-v2, bad-hands-5, ng_deepnegative_v1_75t, bad_prompt


"""

