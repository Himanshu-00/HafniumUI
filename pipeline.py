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
# def generate_image_with_lora(pipeline, prompt, negative_prompt, guidance_scale, num_steps, input_image, num_images=1):
#     try:
#         if not prompt.strip():
#             raise Exception("Please provide a prompt.")

#         print(f"Generating {num_images} images with -  guidance scale: {guidance_scale}, and steps: {num_steps}.")

#         # Convert input image to PIL
#         if isinstance(input_image, np.ndarray):
#             input_image = Image.fromarray(input_image).convert("RGB")
#         elif isinstance(input_image, Image.Image):
#             input_image = input_image.convert("RGB")
#         else:
#             raise Exception("Invalid image format. Please provide a valid image.")

#         # Generate mask once and reuse for all generations
#         mask = segment_and_refine_mask(input_image)
#         generated_images = []

#         for _ in range(num_images):
#             with torch.no_grad():
#                 image = pipeline(
#                     prompt=PROMPT, 
#                     negative_prompt=NPROMPT, 
#                     guidance_scale=guidance_scale,
#                     num_inference_steps=num_steps,
#                     image=input_image,
#                     mask_image=mask
#                 ).images[0]
                
#             generated_images.append(image)

#         print(f"Successfully generated {num_images} images.")
#         return generated_images

#     except Exception as e:
#         raise Exception(f"Error generating images: {e}")
    

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

def generate_image_with_lora(pipeline, prompt, negative_prompt, guidance_scale, num_steps, input_image, progress=gr.Progress()):
    """
    Generate a single image. This function is called for each image we want to generate.
    """
    try:
        # Convert input image to PIL
        if isinstance(input_image, np.ndarray):
            input_image = Image.fromarray(input_image).convert("RGB")
        elif isinstance(input_image, Image.Image):
            input_image = input_image.convert("RGB")
        else:
            raise Exception("Invalid image format. Please provide a valid image.")
        
        # Generate mask
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
            
            return image
    
    except Exception as e:
        raise Exception(f"Error generating image: {e}")

# Load the model with LoRA
pipeline_with_lora = load_model_with_lora()