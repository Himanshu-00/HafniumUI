# # pipeline.py
# import os
# import torch
# from PIL import Image, ImageDraw
# import numpy as np
# from model_loader import load_model_with_lora
# from image_preprocessing import segment_and_refine_mask
# import gradio as gr
# from config import PROMPT, NPROMPT

# # Function to generate an image using the model with LoRA
# def generate_image_with_lora(pipeline, guidance_scale, num_steps, input_image):
#     try:

#         print(f"Generating images with -  guidance scale: {guidance_scale}, and steps: {num_steps}.")

#         # Convert input image to PIL
#         if isinstance(input_image, np.ndarray):
#             input_image = Image.fromarray(input_image).convert("RGB")
#         elif isinstance(input_image, Image.Image):
#             input_image = input_image.convert("RGB")
#         else:
#             raise Exception("Invalid image format. Please provide a valid image.")

#         # Generate mask once and reuse for all generations
#         mask = segment_and_refine_mask(input_image)

#         with torch.no_grad():
#             image = pipeline(
#                 prompt=PROMPT, 
#                 negative_prompt=NPROMPT, 
#                 guidance_scale=guidance_scale,
#                 num_inference_steps=num_steps,
#                 image=input_image,
#                 mask_image=mask
#             ).images[0]
            
#         print(f"Successfully generated images.")    
#         return image      

#     except Exception as e:
#         raise Exception(f"Error generating images: {e}")
    
# # Function to generate images one by one and update gallery
# def generate_images(color, gs, steps, img, num_outputs, progress=gr.Progress(track_tqdm=True)):
#     yield []
#     current_images = []  # Start fresh every time

#     for i in progress.tqdm(range(num_outputs)):
#         progress(i/num_outputs, f"Generating image {i+1}/{num_outputs}")
        
#         new_image = generate_image_with_lora(
#             pipeline_with_lora,
#             guidance_scale=gs,
#             num_steps=steps,
#             input_image=img
#         )
        
#         current_images.append((new_image, f"Generated Image {i+1}/{num_outputs}"))
#         yield current_images


# # Load the model with LoRA
# pipeline_with_lora = load_model_with_lora()


# pipeline.py
import os
import torch
from PIL import Image
import numpy as np
from model_loader import load_model_with_lora
from image_preprocessing import segment_and_refine_mask
import gradio as gr
from config import PROMPT, NPROMPT
import config

# Upscaler imports
from diffusers import StableDiffusionLatentUpscalePipeline

# Initialize upscaler on CPU
def initialize_upscaler():
    return StableDiffusionLatentUpscalePipeline.from_pretrained(
        config.UPSCALER_MODEL,
        torch_dtype=torch.float32  # CPU compatible dtype
    ).to("cpu")

def process_with_upscaler(pipeline, image, target_size):
    original_width, original_height = image.size
    target_width, target_height = target_size
    
    # Calculate scale factors
    width_ratio = target_width / original_width
    height_ratio = target_height / original_height
    scale_factor = max(width_ratio, height_ratio)
    
    if scale_factor <= 1:
        return image.resize(target_size, Image.LANCZOS)
    
    # Multi-step upscaling with CPU-HPU handoff
    current_image = image
    while current_image.width < target_width or current_image.height < target_height:
        # Calculate current scale needed
        current_scale = min(2, 
                          target_width / current_image.width, 
                          target_height / current_image.height)
        
        # Convert to tensor and move to CPU
        with torch.no_grad():
            current_image = pipeline(
                prompt="",
                image=current_image,
                num_inference_steps=20,
                guidance_scale=0,
                noise_level=20,
            ).images[0]
            
    return current_image.resize(target_size, Image.LANCZOS)

def generate_image_with_lora(pipeline, guidance_scale, num_steps, input_image):
    try:
        # Convert and validate input
        if isinstance(input_image, np.ndarray):
            input_image = Image.fromarray(input_image).convert("RGB")
        elif not isinstance(input_image, Image.Image):
            raise ValueError("Invalid image format")

        original_size = input_image.size
        input_image = input_image.resize((1024, 1024))  # SDXL requires 1024x1024

        # Generate mask
        mask = segment_and_refine_mask(input_image)

        # Main generation on GPU
        with torch.no_grad():
            generated_image = pipeline(
                prompt=PROMPT,
                negative_prompt=NPROMPT,
                guidance_scale=guidance_scale,
                num_inference_steps=num_steps,
                image=input_image,
                mask_image=mask
            ).images[0]

        # Post-processing on CPU
        with torch.cpu.amp.autocast(enabled=False):  # Disable mixed precision for CPU
            final_image = process_with_upscaler(
                upscaler_pipeline,
                generated_image,
                original_size
            )

        return final_image

    except Exception as e:
        raise RuntimeError(f"Image generation failed: {e}") from e

# Initialize pipelines
upscaler_pipeline = initialize_upscaler()
pipeline_with_lora = load_model_with_lora()

def generate_images(color, gs, steps, img, num_outputs, progress=gr.Progress()):
    current_images = []
    for i in progress.tqdm(range(num_outputs), desc="Generating"):
        try:
            new_image = generate_image_with_lora(
                pipeline_with_lora,
                guidance_scale=gs,
                num_steps=steps,
                input_image=img
            )
            current_images.append((new_image, f"Image {i+1}/{num_outputs}"))
            yield current_images
        except Exception as e:
            gr.Error(f"Generation failed: {str(e)}")
            yield current_images