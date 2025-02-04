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

# Function to generate an image using the model with LoRA
def generate_image_with_lora(pipeline, guidance_scale, num_steps, input_image, user_mask=None):
    try:
        print(f"Generating images with - guidance scale: {guidance_scale}, and steps: {num_steps}.")

        # Convert input image to PIL
        if isinstance(input_image, np.ndarray):
            input_image = Image.fromarray(input_image).convert("RGB")
        elif isinstance(input_image, Image.Image):
            input_image = input_image.convert("RGB")
        else:
            raise Exception("Invalid image format. Please provide a valid image.")

        # If user_mask is provided, use it; otherwise, generate mask using existing logic
        if user_mask is not None:
            mask = user_mask
            print("Using user-provided mask.")
        else:
            mask = segment_and_refine_mask(input_image)
            print("Using auto-generated mask.")

        # Save the mask for debugging
        debug_dir = "debug_masks"
        os.makedirs(debug_dir, exist_ok=True)
        mask_path = os.path.join(debug_dir, "user_mask.png" if user_mask is not None else "auto_mask.png")
        mask.save(mask_path)
        print(f"Mask saved to {mask_path}")

        with torch.no_grad():
            image = pipeline(
                prompt=PROMPT, 
                negative_prompt=NPROMPT, 
                guidance_scale=guidance_scale,
                num_inference_steps=num_steps,
                image=input_image,
                mask_image=mask
            ).images[0]
            
        print("Successfully generated images.")    
        return image      

    except Exception as e:
        raise Exception(f"Error generating images: {e}")
    
# Function to generate images
def generate_images(color, gs, steps, img, num_outputs):
    current_images = []  # Start fresh every time

    # Extract the user-drawn mask if provided
    user_mask = None
    if isinstance(img, dict) and "mask" in img:
        user_mask = img["mask"].convert("L")  # Convert to grayscale
        img = img["image"]  # Extract the original image

    for i in range(num_outputs):
        print(f"Generating image {i+1}/{num_outputs}")
        
        new_image = generate_image_with_lora(
            pipeline_with_lora,
            guidance_scale=gs,
            num_steps=steps,
            input_image=img,
            user_mask=user_mask
        )
        
        current_images.append((new_image, f"Generated Image {i+1}/{num_outputs}"))

    return current_images


# Load the model with LoRA
pipeline_with_lora = load_model_with_lora()