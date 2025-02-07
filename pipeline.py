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

def generate_image_with_lora(pipeline, guidance_scale, num_steps, input_image):
    try:
        print(f"Generating images with guidance scale: {guidance_scale} and steps: {num_steps}.")

        # If the input comes from Gradio's ImageMask, it will be a dict containing keys like "background" and "layers".
        if isinstance(input_image, dict):
            bg = input_image.get("background", None)
            if bg is None:
                raise Exception("No background image found in the input.")
            # Convert the background to a PIL Image if necessary.
            if isinstance(bg, np.ndarray):
                bg = Image.fromarray(bg).convert("RGB")
            elif isinstance(bg, Image.Image):
                bg = bg.convert("RGB")
            
            # Check for an annotated mask.
            if "layers" in input_image and len(input_image["layers"]) > 0:
                mask_layer = input_image["layers"][0]
                # If the mask layer has an alpha channel and at least one pixel is drawn (alpha > 0), use it.
                if mask_layer.shape[-1] == 4 and mask_layer[:,:,3].max() > 0:
                    mask = Image.fromarray(mask_layer)
                else:
                    mask = segment_and_refine_mask(bg)
            else:
                mask = segment_and_refine_mask(bg)
            input_image = bg  # Use the extracted background image.
        else:
            # Otherwise, assume a plain image was provided.
            if isinstance(input_image, np.ndarray):
                input_image = Image.fromarray(input_image).convert("RGB")
            elif isinstance(input_image, Image.Image):
                input_image = input_image.convert("RGB")
            mask = segment_and_refine_mask(input_image)

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
    
# Generator function to produce images one by one and update the gallery.
def generate_images(color, gs, steps, img, num_outputs, progress=gr.Progress(track_tqdm=True)):
    yield []  # Initialize with an empty gallery.
    current_images = []  # Reset for each generation.
    
    for i in progress.tqdm(range(num_outputs)):
        progress(i/num_outputs, f"Generating image {i+1}/{num_outputs}")
        
        new_image = generate_image_with_lora(
            pipeline_with_lora,
            guidance_scale=gs,
            num_steps=steps,
            input_image=img
        )
        
        current_images.append((new_image, f"Generated Image {i+1}/{num_outputs}"))
        yield current_images

# Load the model with LoRA
pipeline_with_lora = load_model_with_lora()
