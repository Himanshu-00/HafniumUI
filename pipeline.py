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
def generate_image_with_lora(pipeline, prompt, negative_prompt, guidance_scale, num_steps, input_image, num_images=1):
    try:
        print(f"Generating image:   guidance scale: {guidance_scale}, and steps: {num_steps}.")


       # Ensure the input image is in PIL format
        if isinstance(input_image, np.ndarray): 
            input_image = Image.fromarray(input_image).convert("RGB")
        elif isinstance(input_image, Image.Image): 
            input_image = input_image.convert("RGB")
        else:
            raise Exception("Invalid image format. Please provide a valid image.")


        # Segment the input image using rembg and YOLO for face detection for once
        mask = segment_and_refine_mask(input_image)
      

        for _ in range(num_images):
            # Use a callback to capture intermediate results
            def callback(step, timestep, latents):
                with torch.no_grad():
                    # Decode the latents to an image
                    image = pipeline.decode_latents(latents)
                    image = pipeline.numpy_to_pil(image)[0]
                    yield image  # Stream intermediate result
                    
                
            # Generate the image using the pipeline
            final_image = pipeline(
                prompt=PROMPT,
                negative_prompt=NPROMPT,
                guidance_scale=guidance_scale,
                num_inference_steps=num_steps,
                image=input_image,
                mask_image=mask,
                callback=callback,  # Pass the callback for streaming
                callback_steps=5  # Yield an image every 5 steps
            ).images[0]
            print("Image Generated Successfully:")
            yield final_image  # Yield the final image

    except Exception as e:
        raise Exception(f"Error generating images: {e}")

# Load the model with LoRA
pipeline_with_lora = load_model_with_lora()
