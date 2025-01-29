# import os
# import torch
# import requests
# from diffusers import DiffusionPipeline  # Assuming you're using DiffusionPipeline from diffusers library
# from safetensors.torch import load_file
# import config


# # Function to download the LoRA model and save it to the specified path
# def download_lora_model(LORA_DOWNLOAD_URL, LORA_PATH):
#     try:
#          # Check if the model file already exists
#         if os.path.exists(LORA_PATH):
#             print(f"Model already exists at {LORA_PATH}. Skipping download.")
#             return

#         # Ensure model directories exist or create them if they don't
#         if not os.path.exists("models/lora"):
#             os.makedirs("models/lora")
            
#         print(f"Downloading LoRA model from {LORA_DOWNLOAD_URL}...")
#         response = requests.get(LORA_DOWNLOAD_URL, stream=True)
#         response.raise_for_status()  # Raise exception for HTTP errors

#         with open(LORA_PATH, "wb") as f:
#             for chunk in response.iter_content(chunk_size=8192):  # Download in chunks
#                 f.write(chunk)

#         print(f"LoRA model downloaded and saved to {LORA_PATH}")
#     except Exception as e:
#         raise Exception(f"Error downloading LoRA model: {e}")


# # Load the model with LoRA weights
# def load_model_with_lora():
#     device = config.DEVICE
#     model_path = config.MODEL_PATH
#     lora_path = config.LORA_PATH
#     lora_download_url = config.LORA_DOWNLOAD_URL  # Ensure this is defined in your config

#     print(config.DEVICE)
#     print("Loading the Diffusion model...")
#     pipeline = DiffusionPipeline.from_pretrained(
#         model_path,
#         torch_dtype=torch.float16
#     )
#     print("Model loaded successfully.")


#     try:
       
#         # Call the download function here
#         download_lora_model(lora_download_url, lora_path)

#         print("Loading LoRA weights from:", lora_path)
#         lora_state_dict = load_file(lora_path)
#         print("LoRA weights loaded successfully.")

#         def add_lora_to_layer(layer_name, base_layer, lora_state_dict, alpha=0.75):
#             down_name = f"lora_unet_{layer_name}_down.weight"
#             up_name = f"lora_unet_{layer_name}_up.weight"

#             if down_name in lora_state_dict and up_name in lora_state_dict:
#                 down_weight = lora_state_dict[down_name].float()
#                 up_weight = lora_state_dict[up_name].float()

#                 if hasattr(base_layer, 'weight'):
#                     print(f"Applying LoRA weights to layer: {layer_name}")
#                     base_layer.weight.data += alpha * torch.mm(down_weight, up_weight)

#         updated_layers = 0
#         for name, module in pipeline.unet.named_modules():
#             if "attn1" in name or "attn2" in name:
#                 for proj in ["to_k", "to_q", "to_v", "to_out.0"]:
#                     layer_name = f"{name.replace('.', '_')}_{proj}"
#                     if hasattr(module, proj):
#                         add_lora_to_layer(layer_name, getattr(module, proj), lora_state_dict)
#                         updated_layers += 1
#             elif "time_emb" in name and hasattr(module, 'weight'):
#                 layer_name = name.replace('.', '_')
#                 add_lora_to_layer(layer_name, module, lora_state_dict)
#                 updated_layers += 1

#         print(f"Total LoRA layers updated: {updated_layers}")
#         pipeline.to(device)
#         print("Model and LoRA weights successfully loaded and moved to device.")
#         return pipeline

#     except Exception as e:
#         raise Exception(f"Error loading model with LoRA: {e}")


# model_loader.py
import os
import torch
import requests
from PIL import Image
import numpy as np
from diffusers import StableDiffusionXLInpaintPipeline, DPMSolverMultistepScheduler
from safetensors.torch import load_file
import config

def download_lora_model(url, path):
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Downloading LoRA from {url}")
    response = requests.get(url)
    with open(path, "wb") as f:
        f.write(response.content)

def load_model_with_lora():
    device = config.DEVICE
    model_path = config.MODEL_PATH
    lora_path = config.LORA_PATH
    lora_url = config.LORA_DOWNLOAD_URL

    # Initialize inpainting-specific pipeline
    pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        variant="inpaint",
        use_safetensors=True,
        add_watermarker=False
    )

    # Configure scheduler for inpainting
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config,
        algorithm_type="sde-dpmsolver++",
        use_karras_sigmas=True,
        lambda_min_clipped=-float("inf"),
        variance_type="learned_range"
    )

    try:
        download_lora_model(lora_url, lora_path)
        lora_weights = load_file(lora_path)

        # Special handling for 4-channel input layer
        input_conv_layer = pipeline.unet.conv_in
        original_weight = input_conv_layer.weight.data.clone()
        
        # Adjust LoRA weights for 4 channels
        lora_down = lora_weights["lora_unet_conv_in_down.weight"]
        lora_up = lora_weights["lora_unet_conv_in_up.weight"]
        lora_combined = torch.mm(lora_up, lora_down)
        
        # Expand to 4 channels (3 original + 1 mask)
        lora_adjusted = torch.cat([
            lora_combined.to(original_weight.dtype),
            torch.zeros_like(original_weight[3:])
        ], dim=0)
        
        # Apply adjusted LoRA weights
        input_conv_layer.weight.data = original_weight + config.LORA_ALPHA * lora_adjusted

        # Apply LoRA to other layers
        for name, module in pipeline.unet.named_modules():
            if "attn" in name and ("to_k" in name or "to_v" in name or "to_q" in name):
                layer_name = name.replace(".", "_")
                lora_down_name = f"lora_unet_{layer_name}_down.weight"
                lora_up_name = f"lora_unet_{layer_name}_up.weight"
                
                if lora_down_name in lora_weights and lora_up_name in lora_weights:
                    lora_down = lora_weights[lora_down_name]
                    lora_up = lora_weights[lora_up_name]
                    lora_combined = torch.mm(lora_up, lora_down)
                    
                    with torch.no_grad():
                        module.weight.data += config.LORA_ALPHA * lora_combined.to(module.weight.dtype)

        # Memory optimizations
        pipeline.enable_model_cpu_offload()
        pipeline.enable_attention_slicing()
        pipeline.enable_vae_slicing()
        
        # Enable memory-efficient attention
        if config.USE_XFORMERS:
            pipeline.enable_xformers_memory_efficient_attention()
        else:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)

        print(f"Loaded inpainting model with LoRA on {device}")
        return pipeline

    except Exception as e:
        raise RuntimeError(f"Model loading failed: {str(e)}")