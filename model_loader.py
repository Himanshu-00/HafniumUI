# model_loader.py

import torch
from diffusers import DiffusionPipeline
from safetensors.torch import load_file
from config import MODEL_PATH, LOADED_LORA_PATH

# Function to load model with LoRA weights
def load_model_with_lora():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        print("Loading the Diffusion model...")
        pipeline = DiffusionPipeline.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16
        )
        print("Model loaded successfully.")

        print("Loading LoRA weights from:", LOADED_LORA_PATH)
        lora_state_dict = load_file(LOADED_LORA_PATH)
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
