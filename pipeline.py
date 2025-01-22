# pipeline.py
import os
import torch
from PIL import Image, ImageDraw
import numpy as np
import gradio as gr
from model_loader import load_model_with_lora
from image_preprocessing import segment_and_refine_mask

# Function to generate an image using the model with LoRA
def generate_image_with_lora(prompt, negative_prompt, guidance_scale, num_steps, input_image, progress=gr.Progress(track_tqdm=True)):
    try:
        # Input validation
        if input_image is None:
            raise ValueError("Please provide an input image")
        
        if not isinstance(input_image, np.ndarray):
            raise ValueError("Invalid input image format")
            
        if input_image.size == 0:
            raise ValueError("Empty input image")
            
        if not prompt.strip():
            raise ValueError("Please provide a prompt")

        print(f"Generating image with prompt: '{prompt}', guidance scale: {guidance_scale}, and steps: {num_steps}.")
        
        # Convert input image to PIL and validate
        try:
            input_image = Image.fromarray(input_image).convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to process input image: {e}")

        # Validate image dimensions
        if input_image.size[0] < 64 or input_image.size[1] < 64:
            raise ValueError("Input image is too small. Minimum size is 64x64 pixels.")

        # Generate mask with error handling
        try:
            mask = segment_and_refine_mask(input_image)
            if mask is None:
                raise ValueError("Failed to generate mask")
        except Exception as e:
            raise ValueError(f"Error in mask generation: {e}")

        # Use the global pipeline
        pipeline = pipeline_with_lora
        if pipeline is None:
            raise ValueError("Pipeline not properly initialized")

        scheduler = pipeline.scheduler
        timesteps = scheduler.timesteps
        progress(0, desc="Preparing...")

        device = pipeline.device
        do_classifier_free_guidance = guidance_scale > 1.0
        
        with torch.no_grad():
            try:
                latents = pipeline.prepare_latents(
                    batch_size=1,
                    num_channels_latents=4,
                    height=1024,
                    width=1024,
                    dtype=pipeline.unet.dtype,
                    device=device,
                    generator=None
                )
            except Exception as e:
                raise ValueError(f"Error preparing latents: {e}")

            try:
                # Prepare mask and masked image
                mask_image = mask.convert("RGB")
                masked_image = input_image.copy()
                
                # Process images with error handling
                masked_image = pipeline.feature_extractor(masked_image, return_tensors="pt").pixel_values
                mask_image = pipeline.feature_extractor(mask_image, return_tensors="pt").pixel_values
                
                if masked_image is None or mask_image is None:
                    raise ValueError("Failed to process images")
                    
                masked_image = masked_image.to(device=device, dtype=pipeline.unet.dtype)
                mask_image = mask_image.to(device=device, dtype=pipeline.unet.dtype)
            except Exception as e:
                raise ValueError(f"Error processing mask and masked image: {e}")

            try:
                text_embeddings = pipeline._encode_prompt(
                    prompt=prompt,
                    device=device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    negative_prompt=negative_prompt
                )
                
                if text_embeddings is None:
                    raise ValueError("Failed to generate text embeddings")
            except Exception as e:
                raise ValueError(f"Error encoding prompt: {e}")

            # Generation loop with error handling
            for i, t in enumerate(progress.tqdm(timesteps, desc="Generating")):
                try:
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                    noise_pred = pipeline.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=text_embeddings,
                        cross_attention_kwargs=None,
                    ).sample

                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    latents = scheduler.step(noise_pred, t, latents).prev_sample

                    if i % 5 == 0 or i == len(timesteps) - 1:
                        with torch.no_grad():
                            # Generate intermediate image with error handling
                            try:
                                image = pipeline.vae.decode(latents / pipeline.vae.config.scaling_factor, return_dict=False)[0]
                                image = pipeline.image_processor.postprocess(image, output_type="pil")[0]
                                
                                if image is None:
                                    raise ValueError("Failed to generate intermediate image")
                                    
                                yield np.array(image)
                            except Exception as e:
                                print(f"Warning: Failed to generate intermediate preview at step {i}: {e}")
                                continue  # Skip this preview but continue generation

                except Exception as e:
                    raise ValueError(f"Error during generation step {i}: {e}")

        print("Image generation completed successfully.")

    except Exception as e:
        print(f"Error in image generation: {e}")
        # Yield a error image or raise the exception
        error_image = Image.new('RGB', (512, 512), color='red')
        draw = ImageDraw.Draw(error_image)
        draw.text((10, 10), f"Error: {str(e)}", fill='white')
        yield np.array(error_image)

# Load the model globally
pipeline_with_lora = load_model_with_lora()
