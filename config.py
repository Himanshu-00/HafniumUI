# config.py

import os

# Define the paths for model and LoRA weights
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths for the Diffusion model and LoRA weights
MODEL_PATH = os.path.join(BASE_DIR, "diffusers/stable-diffusion-xl-1.0-inpainting-0.1")
LORA_PATH = os.path.join(BASE_DIR, "models/lora/sd_xl_offset_example-lora_1.0.safetensors")

# YOLO model path
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "yolov8n-face-lindevs.pt")

# Path for saving debug images
DEBUG_DIR = os.path.join(BASE_DIR, "debug_images")

# Link for downloading LoRA weights
LORA_DOWNLOAD_URL = "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_offset_example-lora_1.0.safetensors"

# Ensure the directories exist
os.makedirs(os.path.dirname(LORA_PATH), exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)
