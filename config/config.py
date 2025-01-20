import os

# Configuration for model paths, LoRA, and other constants
CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "model_path": "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    "lora_path": "Fooocus/models/loras/sd_xl_offset_example-lora_1.0.safetensors",
    "debug_dir": "debug_images",  # Directory for debug images
    "yolo_model_path": "yolov8n-face-lindevs.pt",  # YOLO model for face detection
}
