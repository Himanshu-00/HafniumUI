import os

# Paths for the model, LoRA weights, and other assets
MODEL_PATH = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
LORA_MODEL_PATH = "model/lora/sd_xl_offset_example-lora_1.0.safetensors"
DEBUG_DIR = "debug_images"
YOLO_PATH = "yolov8n-face-lindevs.pt"

# LoRA model download link
LORA_MODEL_URL = "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_offset_example-lora_1.0.safetensors"

# Create debug directory
os.makedirs(DEBUG_DIR, exist_ok=True)
