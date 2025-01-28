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
from PIL import Image, ImageDraw
import numpy as np
from model_loader import load_model_with_lora
from image_preprocessing import segment_and_refine_mask
import gradio as gr
from config import PROMPT, NPROMPT

# Function to generate an image using the model with LoRA
def generate_image_with_lora(pipeline, prompt, negative_prompt, guidance_scale, num_steps, input_image, num_images=1):
    try:
        if not prompt.strip():
            raise Exception("Please provide a prompt.")

        print(f"Generating {num_images} images with -  guidance scale: {guidance_scale}, and steps: {num_steps}.")

        # Convert input image to PIL
        if isinstance(input_image, np.ndarray):
            input_image = Image.fromarray(input_image).convert("RGB")
        elif isinstance(input_image, Image.Image):
            input_image = input_image.convert("RGB")
        else:
            raise Exception("Invalid image format. Please provide a valid image.")

        # Generate mask once and reuse for all generations
        # Initialize results array with placeholder images
        results = [Image.new("RGB", (512, 512), (40, 40, 40))] * num_images
        mask = segment_and_refine_mask(input_image)
        
        # Track generation states
        current_gen = 0
        yield results  # Initial placeholder
        
        for gen_idx in range(num_images):
            # Prepare for new generation
            results[gen_idx] = Image.new("RGB", (512, 512), (40, 40, 40))
            yield results  # Update specific slot
            
            # Generation logic
            latents = pipeline.prepare_latents(
                image=input_image,
                mask_image=mask,
                width=input_image.width,
                height=input_image.height
            )
            
            scheduler = pipeline.scheduler
            scheduler.set_timesteps(num_steps)
            
            for i, t in enumerate(scheduler.timesteps):
                # Diffusion steps (existing code)
                # ...
                
                if i % 5 == 0 or i == num_steps - 1:
                    # Update current generation preview
                    preview_img = pipeline.numpy_to_pil(latents)[0]
                    results[gen_idx] = preview_img.resize((512, 512))
                    yield results  # Full array update
                    
            # Final image
            final_img = pipeline.numpy_to_pil(latents)[0]
            results[gen_idx] = final_img
            yield results
            
        return results

    except Exception as e:
        raise gr.Error(f"Generation failed: {str(e)}")


def pipe_callback(step, timestep, latents, pipeline, generated_images, idx):
    # Convert latents to image every few steps
    if step % 5 == 0:  # Update every 5 steps
        with torch.no_grad():
            image = pipeline.decode_latents(latents)
            image = pipeline.numpy_to_pil(image)[0]
            
        if len(generated_images) <= idx:
            generated_images.append(image)
        else:
            generated_images[idx] = image


# Load the model with LoRA
pipeline_with_lora = load_model_with_lora()
