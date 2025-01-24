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

A photorealistic portrait of a 23-year-old male model from Canada, exuding professionalism in a tailored navy suit and a tie.
He stands confidently against a sleek, light gray backdrop inside a modern photo studio, illuminated by soft, diffused lighting that enhances the suit's texture and fit. 
The image is captured in a medium shot, 8K HDR, showcasing the refined elegance of his attire.

"""
NPROMPT = """

"No distracting visual elements or cluttered backgrounds with complex patterns or busy textures." 
"No low-resolution, pixelated, blurry, or out-of-focus images lacking sharp details." 
"No stylized, cartoon-like, illustrated, painterly, or non-photorealistic rendering styles." 
"No exaggerated facial expressions, dramatic makeup, extreme contouring, or overly styled hair." 
"No informal, wrinkled, ill-fitting, or inappropriate clothing; avoid casual wear, visible logos, or distracting accessories." "No harsh, uneven, or extreme lighting conditions causing overexposure, harsh shadows, or lens flares." 
"No environmental context or contextual backgrounds; strictly controlled studio-like setting." 
"No perspective distortions, unflattering camera angles, or unintended body proportions."
"No heavy post-processing, unnatural color grading, or obvious digital manipulation." 
"No dynamic or candid poses; maintain a neutral, professional, and controlled posture." 
"No facial orientation other than direct, centered front-facing view with neutral expression."
"No neckwear with bold patterns, reflective surfaces, or unconventional designs."
"No unnatural hand positioning or gestures that appear forced or awkward."
"No decorative or alternative neckwear styles."
"Maintain precise, anatomically correct body proportions with natural shoulder width and limb length."
"No visual stretching or compression of body parts, ensuring natural human silhouette." "Preserve natural, upright body alignment without extreme inclinations or unnatural postures."
"Include a conservative, solid-color necktie in navy blue or charcoal gray" "Position the tie centrally, neatly knotted at the collar" "Ensure tie width is proportional to suit lapels" "Tie should have a matte, non-reflective finish"



""" 
GS = 7,5
STEPS = 50