# pipeline.py
import os
import torch
from PIL import Image, ImageDraw
import numpy as np
from model_loader import load_model_with_lora
from image_preprocessing import segment_and_refine_mask
from config import PROMPT, NPROMPT

# Function to generate an image using the model with LoRA
def generate_image_with_lora(pipeline, prompt, negative_prompt, guidance_scale, num_steps, input_image):
    try:
        if not prompt.strip():
            raise ValueError("Please provide a valid prompt.")
        
        print(f"Generating image with prompt: '{prompt}', negative prompt: '{negative_prompt}', guidance scale: {guidance_scale}, steps: {num_steps}.")
        
        # Convert input image to PIL format
        if not isinstance(input_image, np.ndarray):
            raise TypeError("Input image must be a NumPy array.")
        input_image = Image.fromarray(input_image).convert("RGB")
        print("Input image converted to PIL format.")

        # Generate segmentation mask
        mask = segment_and_refine_mask(input_image)
        if mask is None or not isinstance(mask, Image.Image):
            raise ValueError("Segmentation mask is invalid or not generated.")
        print("Mask generated successfully.")

        # Generate the image using the pipeline
        with torch.no_grad():
            output = pipeline(
                prompt=PROMPT,
                negative_prompt=NPROMPT,
                guidance_scale=guidance_scale,
                num_inference_steps=num_steps,
                image=input_image,
                mask_image=mask
            )
        
        if not hasattr(output, "images") or not output.images:
            raise ValueError("Pipeline did not return any images.")
        
        image = output.images[0]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.array(image))  # Ensure valid PIL.Image
        
        print("Image generated successfully.")
        return image

    except Exception as e:
        print(f"Error generating image: {e}")
        raise

# Load the model with LoRA
pipeline_with_lora = load_model_with_lora()