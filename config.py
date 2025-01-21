import os
import torch

# Configuration for model paths, LoRA, and other constants
CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",  # Automatically choose CUDA or CPU
    "model_path": "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",  # Path to the base model
    "lora_url": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_offset_example-lora_1.0.safetensors",  # URL for LoRA weights
    "lora_path": "models/lora/sd_xl_offset_example-lora_1.0.safetensors",  # Local path to save LoRA weights
    "debug_dir": "debug_images",  # Directory for debug images
    "guidance_scale" : 7.5,
    "steps" : 30,
    "yolo_model_path": "yolov8n-face-lindevs.pt",  # YOLO model for face detection
}
