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


# # pipeline.py
# import os
# import torch
# from PIL import Image
# import numpy as np
# from model_loader import load_model_with_lora
# from image_preprocessing import segment_and_refine_mask
# import gradio as gr
# from config import PROMPT, NPROMPT

# def generate_image_with_lora(pipeline, prompt, negative_prompt, guidance_scale, num_steps, input_image, progress=gr.Progress()):
#     """
#     Generate a single image. This function is called for each image we want to generate.
#     """
#     try:
#         # Convert input image to PIL
#         if isinstance(input_image, np.ndarray):
#             input_image = Image.fromarray(input_image).convert("RGB")
#         elif isinstance(input_image, Image.Image):
#             input_image = input_image.convert("RGB")
#         else:
#             raise Exception("Invalid image format. Please provide a valid image.")
        
#         # Generate mask
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
            
#             return image
    
#     except Exception as e:
#         raise Exception(f"Error generating image: {e}")

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
from diffusers import DiffusionPipeline

def preview_callback(pipe, step_index, timestep, callback_kwargs):
    latents = callback_kwargs["latents"]  # Get latents from callback_kwargs
    with torch.no_grad():
        # Convert latents to image
        latents_copy = 1 / 0.18215 * latents
        image = pipe.vae.decode(latents_copy).sample  # Use 'pipe' here, not 'pipeline'
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = (image[0] * 255).astype(np.uint8)
        callback_data["current_image"] = Image.fromarray(image)
    
    return callback_kwargs  # Must return modified callback_kwargs


def generate_image_with_lora(pipeline, prompt, negative_prompt, guidance_scale, num_steps, input_image, progress=gr.Progress()):
    try:
        # Prepare the image
        if isinstance(input_image, np.ndarray):
            input_image = Image.fromarray(input_image).convert("RGB")
        elif isinstance(input_image, Image.Image):
            input_image = input_image.convert("RGB")
        else:
            raise ValueError("Invalid image format. Please provide a valid image.")

        mask = segment_and_refine_mask(input_image)
        yield [(input_image, "Starting generation...")]

        callback_data = {"current_image": None}

        result = pipeline(
            prompt=prompt,
            image=input_image,
            mask_image=mask,
            negative_prompt=negative_prompt,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=torch.Generator(device=pipeline.device),
            callback_on_step_end=preview_callback,
            callback_on_step_end_tensor_inputs=["latents"]
        )

        # Yield intermediate images to update progress
        for step in range(num_steps):
            progress(step / num_steps, f"Step {step + 1}/{num_steps}")
            if callback_data["current_image"]:
                yield [(callback_data["current_image"], f"Step {step + 1}/{num_steps}")]

        yield [(result.images[0], "Final Result")]

    except Exception as e:
        print(f"[Error] {str(e)}")
        raise gr.Error(f"Image generation failed: {str(e)}")


# Load the model with LoRA
pipeline_with_lora = load_model_with_lora()