# pipeline.py
import os
import torch
from PIL import Image, ImageDraw
import numpy as np
from model_loader import load_model_with_lora
from image_preprocessing import segment_and_refine_mask

def generate_image_with_lora(pipeline, prompt, negative_prompt, guidance_scale, num_steps, input_image, progress=None):
    try:
        if not prompt.strip():
            raise Exception("Please provide a prompt.")
        
        print(f"Generating image with prompt: '{prompt}', negative prompt: '{negative_prompt}', guidance scale: {guidance_scale}, and steps: {num_steps}.")
        input_image = Image.fromarray(input_image).convert("RGB")
        
        # Segment the input image using rembg and YOLO for face detection
        mask = segment_and_refine_mask(input_image)
        
        # Define callback for updating progress
        def callback_fn(step: int, timestep: int, latents: torch.FloatTensor):
            if progress is not None:
                # Convert latents to image
                with torch.no_grad():
                    latents = 1 / 0.18215 * latents
                    image = pipeline.vae.decode(latents).sample
                    image = (image / 2 + 0.5).clamp(0, 1)
                    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
                    image = (image * 255).round().astype("uint8")[0]
                    image = Image.fromarray(image)
                
                # Update progress
                progress(image, step / num_steps)
            return True
        
        with torch.no_grad():
            # Generate the image using the mask created from segmentation and YOLO
            image = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_steps,
                image=input_image,
                mask_image=mask,
                callback=callback_fn,
                callback_steps=1
            ).images[0]
        
        print("Image generated successfully.")
        return image
    
    except Exception as e:
        raise Exception(f"Error generating image: {e}")

# Load the model with LoRA
pipeline_with_lora = load_model_with_lora()

