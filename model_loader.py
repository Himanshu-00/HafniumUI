import os
import torch
import requests
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler  # Assuming you're using DiffusionPipeline from diffusers library
from safetensors.torch import load_file
import config


# Function to download the LoRA model and save it to the specified path
def download_lora_model(LORA_DOWNLOAD_URL, LORA_PATH):
    try:
         # Check if the model file already exists
        if os.path.exists(LORA_PATH):
            print(f"Model already exists at {LORA_PATH}. Skipping download.")
            return

        # Ensure model directories exist or create them if they don't
        if not os.path.exists("models/lora"):
            os.makedirs("models/lora")
            
        print(f"Downloading LoRA model from {LORA_DOWNLOAD_URL}...")
        response = requests.get(LORA_DOWNLOAD_URL, stream=True)
        response.raise_for_status()  # Raise exception for HTTP errors

        with open(LORA_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):  # Download in chunks
                f.write(chunk)

        print(f"LoRA model downloaded and saved to {LORA_PATH}")
    except Exception as e:
        raise Exception(f"Error downloading LoRA model: {e}")


# Load the model with LoRA weights
def load_model_with_lora():
    device = config.DEVICE
    model_path = config.MODEL_PATH
    lora_path = config.LORA_PATH
    lora_download_url = config.LORA_DOWNLOAD_URL  # Ensure this is defined in your config

    print(config.DEVICE)
    print("Loading the Diffusion model...")
    pipeline = DiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16
    )

    # Best configuration for realistic inpainting (SDXL optimized)
    scheduler_config = pipeline.scheduler.config

    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        scheduler_config,
        algorithm_type="sde-dpmsolver++",  # Changed to SDE variant for better detail
        solver_order=2,  # Reduced for better VRAM/memory balance
        use_karras_sigmas=True,
        lower_order_final=True,
        solver_type="midpoint",  # Better for inpainting than heun
        prediction_type="epsilon",  # Keep for SDXL base
        thresholding=True,
        dynamic_thresholding_ratio=0.90,  # Lowered to prevent artifacts
        sample_max_value=1.0,
        steps_offset=1,
        timestep_spacing="trailing",  # Keep trailing for inpainting
        # New parameters for SDXL inpainting
        lambda_min_clipped=-float("inf"),  # Better mask blending
        variance_type="learned_range"  # Match SDXL training
    )

    print("Model loaded successfully.")


    try:
       
        # Call the download function here
        download_lora_model(lora_download_url, lora_path)

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
