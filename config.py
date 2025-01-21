# config.py

import os

# Model and weights paths
MODEL_PATH = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
LOADED_LORA_PATH = "model/lora/sd_xl_offset_example-lora_1.0.safetensors"
DEBUG_DIR = "debug_images"

# Ensure the directories exist
os.makedirs(DEBUG_DIR, exist_ok=True)

# Paths for downloaded models
LOADER_MODEL_URL = "https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/model.ckpt"  # Example URL
