import os
import torch
from diffusers import DiffusionPipeline
from safetensors.torch import load_file
from config import CONFIG
import requests

def download_lora_weights(url, save_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded LoRA weights to: {save_path}")
    else:
        raise Exception(f"Failed to download LoRA weights from {url}. Status code: {response.status_code}")

def load_model_with_lora():
    device = CONFIG["device"]

    try:
        # Ensure the directory for the LoRA path exists
        lora_dir = os.path.dirname(CONFIG["lora_path"])
        if not os.path.exists(lora_dir):
            os.makedirs(lora_dir, exist_ok=True)
            print(f"Created directory for LoRA weights: {lora_dir}")

        # Check if LoRA weights file exists; if not, download it
        if not os.path.isfile(CONFIG["lora_path"]):
            print("LoRA weights not found, downloading...")
            download_lora_weights(CONFIG["lora_url"], CONFIG["lora_path"])

        print("Loading the Diffusion model...")
        pipeline = DiffusionPipeline.from_pretrained(CONFIG["model_path"], torch_dtype=torch.float16)
        print("Model loaded successfully.")

        print("Loading LoRA weights from:", CONFIG["lora_path"])
        lora_state_dict = load_file(CONFIG["lora_path"])
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
