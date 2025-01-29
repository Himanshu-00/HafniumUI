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
def generate_image_with_lora(pipeline, guidance_scale, num_steps, input_image, progress=gr.Progress()):
    try:

        print(f"Generating images with -  guidance scale: {guidance_scale}, and steps: {num_steps}.")

        # Convert input image to PIL
        if isinstance(input_image, np.ndarray):
            input_image = Image.fromarray(input_image).convert("RGB")
        elif isinstance(input_image, Image.Image):
            input_image = input_image.convert("RGB")
        else:
            raise Exception("Invalid image format. Please provide a valid image.")

        # Generate mask once and reuse for all generations
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
            
        print(f"Successfully generated images.")    
        return image      

    except Exception as e:
        raise Exception(f"Error generating images: {e}")
    
# Function to generate images one by one and update gallery
def generate_images(color, gs, steps, img, num_outputs, current_state, progress=gr.Progress(track_tqdm=True)):
    progress = gr.Progress()
    # Clear the gallery if we're starting a new generation
    current_images = current_state

    for i in progress.tqdm(range(num_outputs)):
        progress(i/num_outputs, f"Generating image {i+1}/{num_outputs}")
        
        # Generate new image
        new_image = generate_image_with_lora(
        pipeline_with_lora,
        guidance_scale=gs,
        num_steps=steps,
        input_image=img
        )
        
        # Add to our list with a caption
        current_images.append((new_image, f"Generated Image {i+1}/{num_outputs}"))
        # Yield current state for gallery update
        yield current_images, f"Image {i + 1} Generated"
        

# Load the model with LoRA
pipeline_with_lora = load_model_with_lora()
