import os
import torch
import requests
from diffusers import DiffusionPipeline
from safetensors.torch import load_file
from ultralytics import YOLO
from config import CONFIG  # Importing config for paths

# Function to download LoRA weights from the URL
def download_lora_weights(url, save_path):
    print(f"Downloading LoRA weights from: {url}")
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, "wb") as f:
            f.write(response.content)
        print(f"LoRA weights downloaded and saved to {save_path}")
    else:
        raise Exception(f"Failed to download LoRA weights. Status code: {response.status_code}")

# Load the model with LoRA weights
def load_model_with_lora():
    device = CONFIG['device']
    model_path = CONFIG['model_path']
    lora_path = CONFIG['lora_path']
    lora_url = CONFIG['lora_url']

    # If LoRA weights are not downloaded, download them
    if not os.path.exists(lora_path):
        download_lora_weights(lora_url, lora_path)

    try:
        print("Loading the Diffusion model...")
        pipeline = DiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16
        )
        print("Model loaded successfully.")

        print("Loading LoRA weights from:", lora_path)
        lora_state_dict = load_file(lora_path)
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
    yolo_model = YOLO(CONFIG['yolo_model_path'])
    return yolo_model
