# pipeline.py
import torch
from PIL import Image
import numpy as np
from model_loader import load_model_with_lora
from image_preprocessing import segment_and_refine_mask
import gradio as gr
from config import PROMPT, NPROMPT

def generate_image_with_lora(pipeline, guidance_scale, num_steps, input_image):
    """Generate a single image using the model with LoRA"""
    try:
        print(f"Generating image with guidance scale: {guidance_scale}, steps: {num_steps}")

        # Convert input image to PIL
        if isinstance(input_image, np.ndarray):
            input_image = Image.fromarray(input_image).convert("RGB")
        elif isinstance(input_image, Image.Image):
            input_image = input_image.convert("RGB")
        else:
            raise ValueError("Invalid image format. Please provide a valid image.")

        # Generate mask
        mask = segment_and_refine_mask(input_image)

        # Generate image
        with torch.no_grad():
            result = pipeline(
                prompt=PROMPT,
                negative_prompt=NPROMPT,
                guidance_scale=guidance_scale,
                num_inference_steps=num_steps,
                image=input_image,
                mask_image=mask
            ).images[0]

        print("Successfully generated image.")
        return result

    except Exception as e:
        raise Exception(f"Error generating image: {str(e)}")

def generate_images(color, gs, steps, img, num_outputs, current_state, progress=gr.Progress(track_tqdm=True)):
    """Generate multiple images and update the gallery progressively"""
    try:
        current_images = []
        
        for i in progress.tqdm(range(num_outputs), desc="Generating images"):
            # Use tqdm for overall progress tracking
            progress.update(desc=f"Generating image {i+1}/{num_outputs}")
            
            new_image = generate_image_with_lora(
                pipeline_with_lora,
                guidance_scale=gs,
                num_steps=steps,
                input_image=img
            )
            
            current_images.append((new_image, f"Generated Image {i+1}/{num_outputs}"))
            yield current_images, current_images

    except Exception as e:
        print(f"Error in generate_images: {str(e)}")
        raise

# Initialize the pipeline
pipeline_with_lora = load_model_with_lora()