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

def generate_image_with_lora(pipeline, prompt, negative_prompt, guidance_scale, num_steps, input_image, progress=gr.Progress()):
    """Generate an image with live step updates using callback_on_step_end."""
    try:
        if isinstance(input_image, np.ndarray):
            input_image = Image.fromarray(input_image).convert("RGB")
        elif isinstance(input_image, Image.Image):
            input_image = input_image.convert("RGB")
        else:
            raise ValueError("Invalid image format. Please provide a valid image.")

        mask = segment_and_refine_mask(input_image)

        # **1️⃣ Yield Input Image Immediately**
        yield [(input_image, "Starting generation...")]

        # **2️⃣ Set Up Callback for Live Updates**
        def preview_callback(step: int, timestep: int, latents: torch.FloatTensor, **kwargs):
            """Updates UI with intermediate images at each step."""
            progress_percentage = (step / num_steps) * 100
            progress(step / num_steps, desc=f"Step {step}/{num_steps} ({progress_percentage:.1f}%)")

            # Convert latents to image correctly
            with torch.no_grad():
                latents_copy = latents.clone().detach()
                latents_copy = 1 / 0.18215 * latents_copy
                image_tensor = pipeline.vae.decode(latents_copy).sample
                image_tensor = (image_tensor / 2 + 0.5).clamp(0, 1)
                image_np = image_tensor.cpu().permute(0, 2, 3, 1).numpy()
                image_np = (image_np * 255).round().astype("uint8")
                intermediate_image = Image.fromarray(image_np[0])

                print(f"[Console Log] Step {step}/{num_steps} - Intermediate image updated")

                # **Yield Intermediate Image**
                yield [(intermediate_image, f"Step {step}/{num_steps}")]

        # **4️⃣ Set Up Random Generator**
        generator = torch.Generator(device=pipeline.device)

        # **5️⃣ Run the Pipeline with `callback_on_step_end`**
        result = pipeline(
            prompt=prompt,
            image=input_image,
            mask_image=mask,
            negative_prompt=negative_prompt,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            callback_on_step_end=preview_callback  # 🔥 Correct way to do step updates
        )

        # **6️⃣ Ensure Final Image is Displayed**
        yield [(result.images[0], "Final Result")]

    except Exception as e:
        print(f"[Error] {str(e)}")
        raise ValueError(f"Error generating image: {e}")

# Load the model with LoRA
pipeline_with_lora = load_model_with_lora()