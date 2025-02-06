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
from PIL import Image, ImageDraw
import numpy as np
from model_loader import load_model_with_lora
from image_preprocessing import segment_and_refine_mask
import gradio as gr
from config import PROMPT, NPROMPT

# Function to generate an image using the model with LoRA
def generate_image_with_lora(pipeline, guidance_scale, num_steps, input_image, user_mask=None):
    try:
        print(f"Generating images with - guidance scale: {guidance_scale}, steps: {num_steps}")

        # Determine the mask to use
        if user_mask is not None:
            mask_array = np.array(user_mask)
            if np.any(mask_array > 0):
                print("Using user-provided mask")
                mask = user_mask
            else:
                print("User mask empty; using default")
                mask = segment_and_refine_mask(input_image)
        else:
            print("No user mask; using default segmentation")
            mask = segment_and_refine_mask(input_image)

        # Run the pipeline
        with torch.no_grad():
            image = pipeline(
                prompt=PROMPT, 
                negative_prompt=NPROMPT, 
                guidance_scale=guidance_scale,
                num_inference_steps=num_steps,
                image=input_image,
                mask_image=mask
            ).images[0]
            
        return image      

    except Exception as e:
        raise Exception(f"Error generating images: {e}")
    
    
# Function to generate images one by one and update gallery
def generate_images(color, gs, steps, img, num_outputs, progress=gr.Progress(track_tqdm=True)):
    yield []
    current_images = []

    # Unpack the image and mask from ImageMask component
    input_image_pil, user_mask_np = img

    # Convert input image to PIL
    if isinstance(input_image_pil, np.ndarray):
        input_image = Image.fromarray(input_image_pil).convert("RGB")
    else:
        input_image = input_image_pil.convert("RGB")

    # Process user-drawn mask
    user_mask = None
    if user_mask_np is not None:
        if np.any(user_mask_np):
            user_mask = Image.fromarray(user_mask_np).convert('L')  # Convert to grayscale PIL

    # Generate each image
    for i in progress.tqdm(range(num_outputs)):
        progress(i/num_outputs, f"Generating image {i+1}/{num_outputs}")
        new_image = generate_image_with_lora(
            pipeline_with_lora,
            guidance_scale=gs,
            num_steps=steps,
            input_image=input_image,
            user_mask=user_mask  # Pass the user's mask
        )
        current_images.append((new_image, f"Generated Image {i+1}/{num_outputs}"))
        yield current_images

# Load the model with LoRA
pipeline_with_lora = load_model_with_lora()
