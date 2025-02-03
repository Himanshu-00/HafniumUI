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
import torch
from PIL import Image
import numpy as np
from model_loader import load_model_with_lora
from image_preprocessing import segment_and_refine_mask
import gradio as gr
from config import PROMPT, NPROMPT

def get_mask(input_image, mask_mode):
    """Get mask based on selected mode"""
    if mask_mode == "Automatic Masking":
        return segment_and_refine_mask(input_image)
    else:
        # For manual masking, extract the mask from the sketch layer
        if isinstance(input_image, dict) and "mask" in input_image:
            # Convert the mask to grayscale and invert it
            mask = input_image["mask"].convert("L")
            return Image.eval(mask, lambda x: 255 - x)
        return None

def generate_image_with_lora(pipeline, guidance_scale, num_steps, input_image, mask_mode):
    try:
        print(f"Generating images with - guidance scale: {guidance_scale}, and steps: {num_steps}.")

        # Handle input image processing
        if isinstance(input_image, dict) and "image" in input_image:
            # For manual masking mode, use the base image
            base_image = input_image["image"].convert("RGB")
        elif isinstance(input_image, np.ndarray):
            base_image = Image.fromarray(input_image).convert("RGB")
        elif isinstance(input_image, Image.Image):
            base_image = input_image.convert("RGB")
        else:
            raise Exception("Invalid image format. Please provide a valid image.")

        # Get appropriate mask based on mode
        mask = get_mask(input_image, mask_mode)
        if mask is None:
            raise Exception("No mask available. Please draw a mask or switch to automatic mode.")

        with torch.no_grad():
            image = pipeline(
                prompt=PROMPT,
                negative_prompt=NPROMPT,
                guidance_scale=guidance_scale,
                num_inference_steps=num_steps,
                image=base_image,
                mask_image=mask
            ).images[0]
            
        print(f"Successfully generated images.")    
        return image

    except Exception as e:
        raise Exception(f"Error generating images: {e}")

def generate_images(color, gs, steps, img, num_outputs, mask_mode, progress=gr.Progress(track_tqdm=True)):
    yield []
    current_images = []

    for i in progress.tqdm(range(num_outputs)):
        progress(i/num_outputs, f"Generating image {i+1}/{num_outputs}")
        
        new_image = generate_image_with_lora(
            pipeline_with_lora,
            guidance_scale=gs,
            num_steps=steps,
            input_image=img,
            mask_mode=mask_mode
        )
        
        current_images.append((new_image, f"Generated Image {i+1}/{num_outputs}"))
        yield current_images

# Load the model with LoRA
pipeline_with_lora = load_model_with_lora()