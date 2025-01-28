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
#         print(f"Generating image:   guidance scale: {guidance_scale}, and steps: {num_steps}.")


#        # Ensure the input image is in PIL format
#         if isinstance(input_image, np.ndarray): 
#             input_image = Image.fromarray(input_image).convert("RGB")
#         elif isinstance(input_image, Image.Image): 
#             input_image = input_image.convert("RGB")
#         else:
#             raise Exception("Invalid image format. Please provide a valid image.")


#         # Segment the input image using rembg and YOLO for face detection for once
#         mask = segment_and_refine_mask(input_image)
      

#         for _ in range(num_images):
#             # Use a callback to capture intermediate results
#             def callback(step, timestep, latents):
#                 with torch.no_grad():
#                     # Decode the latents to an image
#                     image = pipeline.decode_latents(latents)
#                     image = pipeline.numpy_to_pil(image)[0]
#                     yield image  # Stream intermediate result
                    
                
#             # Generate the image using the pipeline
#             final_image = pipeline(
#                 prompt=PROMPT,
#                 negative_prompt=NPROMPT,
#                 guidance_scale=guidance_scale,
#                 num_inference_steps=num_steps,
#                 image=input_image,
#                 mask_image=mask,
#                 callback=callback,  # Pass the callback for streaming
#                 callback_steps=5  # Yield an image every 5 steps
#             ).images[0]
#             print("Image Generated Successfully:")
#             yield final_image  # Yield the final image

#     except Exception as e:
#         raise Exception(f"Error generating images: {e}")

# # Load the model with LoRA
# pipeline_with_lora = load_model_with_lora()

import os
import torch
from PIL import Image
import numpy as np
from model_loader import load_model_with_lora
from image_preprocessing import segment_and_refine_mask
from config import PROMPT, NPROMPT

def validate_and_convert_image(input_image):
    """Validate and convert input image to RGB PIL Image format."""
    if input_image is None:
        raise ValueError("No image provided. Please upload an image.")
        
    try:
        if isinstance(input_image, np.ndarray):
            return Image.fromarray(input_image).convert("RGB")
        elif isinstance(input_image, Image.Image):
            return input_image.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(input_image)}")
    except Exception as e:
        raise ValueError(f"Error converting image: {str(e)}")

def generate_image_with_lora(pipeline, prompt, negative_prompt, guidance_scale, num_steps, input_image, num_images=1):
    """Generate images using the pipeline with LoRA and stream intermediate results."""
    try:
        print(f"Starting image generation with guidance scale: {guidance_scale}, steps: {num_steps}")
        
        # Validate and convert input image
        processed_image = validate_and_convert_image(input_image)
        
        # Generate mask for the processed image
        try:
            mask = segment_and_refine_mask(processed_image)
        except Exception as e:
            raise ValueError(f"Error in mask generation: {str(e)}")

        # Create a list to store intermediate results
        intermediate_images = []

        # Define callback for intermediate results
        def callback(step, timestep, latents):
            with torch.no_grad():
                try:
                    image = pipeline.decode_latents(latents)
                    image = pipeline.numpy_to_pil(image)[0]
                    intermediate_images.append(image)
                except Exception as e:
                    print(f"Warning: Error in callback at step {step}: {str(e)}")

        # Generate images
        for i in range(num_images):
            try:
                # Clear intermediate images for each new generation
                intermediate_images.clear()
                
                # Start the generation process
                result = pipeline(
                    prompt=PROMPT,
                    negative_prompt=NPROMPT,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_steps,
                    image=processed_image,
                    mask_image=mask,
                    callback=callback,
                    callback_steps=5
                )
                
                # Yield intermediate results first
                for img in intermediate_images:
                    yield img
                
                # Finally, yield the completed image
                final_image = result.images[0]
                print(f"Successfully generated image {i+1}/{num_images}")
                yield final_image
                
            except Exception as e:
                raise ValueError(f"Error during image generation (iteration {i+1}): {str(e)}")

    except Exception as e:
        if isinstance(e, ValueError):
            raise e
        raise Exception(f"Unexpected error in image generation pipeline: {str(e)}")

# Initialize the pipeline
pipeline_with_lora = load_model_with_lora()