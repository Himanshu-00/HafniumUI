import os
import torch
from diffusers import DiffusionPipeline
from safetensors.torch import load_file
from ultralytics import YOLO
import requests
from config import MODEL_PATH, LORTA_PATH, YOLO_PATH, LOTA_DOWNLOAD_URL

# Function to download the LoRA model if not already present
def download_lora_model():
    # Check if the 'model/lora' folder exists, and create it if not
    if not os.path.exists(os.path.dirname(LORTA_PATH)):
        print(f"Creating directory {os.path.dirname(LORTA_PATH)}...")
        os.makedirs(os.path.dirname(LORTA_PATH), exist_ok=True)
    
    if not os.path.exists(LORTA_PATH):
        print(f"Downloading LoRA model from {LOTA_DOWNLOAD_URL}...")
        response = requests.get(LOTA_DOWNLOAD_URL, stream=True)
        if response.status_code == 200:
            with open(LORTA_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"LoRA model downloaded and saved to {LORTA_PATH}")
        else:
            raise Exception(f"Failed to download LoRA model. Status code: {response.status_code}")

# Load the model with LoRA weights
def load_model_with_lora():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Download LoRA model if not present
    download_lora_model()

    try:
        print("Loading the Diffusion model...")
        pipeline = DiffusionPipeline.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16
        )
        print("Model loaded successfully.")

        print("Loading LoRA weights from:", LORTA_PATH)
        lora_state_dict = load_file(LORTA_PATH)
        print("LoRA weights loaded successfully.")

        def add_lora_to_layer(layer_name, base_layer, lora_state_dict, alpha=0.75):
            down_name = f"lora_unet_{layer_name}_down.weight"
            up_name = f"lora_unet_{layer_name}_up.weight"

            if down_name in lora_state_dict and up_name in lora_state_dict:
                down_weight = lora_state_dict[down_name].float()
                up_weight = lora_state_dict[up_name].float()

                if hasattr(base_layer, 'weight'):
                    print(f"Applying LoRA weights to layer: {layer_name}")
                    base_layer.weight.data += alpha * torch.mm(down_weight, up_weight)

        updated_layers = 0
        for name, module in pipeline.unet.named_modules():
            if "attn1" in name or "attn2" in name:
                for proj in ["to_k", "to_q", "to_v", "to_out.0"]:
                    layer_name = f"{name.replace('.', '_')}_{proj}"
                    if hasattr(module, proj):
                        add_lora_to_layer(layer_name, getattr(module, proj), lora_state_dict)
                        updated_layers += 1
            elif "time_emb" in name and hasattr(module, 'weight'):
                layer_name = name.replace('.', '_')
                add_lora_to_layer(layer_name, module, lora_state_dict)
                updated_layers += 1

        print(f"Total LoRA layers updated: {updated_layers}")
        pipeline.to(device)
        print("Model and LoRA weights successfully loaded and moved to device.")
        return pipeline

    except Exception as e:
        raise Exception(f"Error loading model with LoRA: {e}")

# Load YOLO face detection model
def load_yolo_model():
    try:
        print("Loading YOLO model...")
        yolo_model = YOLO(YOLO_PATH)
        print("YOLO model loaded successfully.")
        return yolo_model
    except Exception as e:
        raise Exception(f"Error loading YOLO model: {e}")
