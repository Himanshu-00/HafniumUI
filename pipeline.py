# pipeline.py
import os
import torch
from PIL import Image, ImageDraw
import numpy as np
import gradio as gr
from model_loader import load_model_with_lora
from image_preprocessing import segment_and_refine_mask

# Function to generate an image using the model with LoRA
def generate_image_with_lora(pipeline, prompt, negative_prompt, guidance_scale, num_steps, input_image, progress=gr.Progress(track_tqdm=True)):
    try:
        if not prompt.strip():
            raise Exception("Please provide a prompt.")

        print(f"Generating image with prompt: '{prompt}', guidance scale: {guidance_scale}, and steps: {num_steps}.")
        input_image = Image.fromarray(input_image).convert("RGB")
        mask = segment_and_refine_mask(input_image)

        # Enable streaming outputs by accessing the internal scheduler
        scheduler = pipeline.scheduler
        timesteps = scheduler.timesteps
        progress(0, desc="Preparing...")

        # Initialize the latents and other parameters
        device = pipeline.device
        do_classifier_free_guidance = guidance_scale > 1.0
        
        # Get the initial latents and prepare inputs
        with torch.no_grad():
            latents = pipeline.prepare_latents(
                batch_size=1,
                num_channels_latents=4,
                height=1024,
                width=1024,
                dtype=pipeline.unet.dtype,
                device=device,
                generator=None
            )

            # Prepare mask and masked image
            mask_image = mask.convert("RGB")
            masked_image = input_image.copy()
            masked_image = pipeline.feature_extractor(masked_image, return_tensors="pt").pixel_values
            mask_image = pipeline.feature_extractor(mask_image, return_tensors="pt").pixel_values
            masked_image = masked_image.to(device=device, dtype=pipeline.unet.dtype)
            mask_image = mask_image.to(device=device, dtype=pipeline.unet.dtype)

            # Encode prompt
            text_embeddings = pipeline._encode_prompt(
                prompt=prompt,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=do_classifier_free_guidance,
                negative_prompt=negative_prompt
            )

            # Denoising loop with progress updates
            for i, t in enumerate(progress.tqdm(timesteps, desc="Generating")):
                # Expand latents for classifier-free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                # Predict noise residual
                noise_pred = pipeline.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings,
                    cross_attention_kwargs=None,
                ).sample

                # Perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # Compute previous noisy sample
                latents = scheduler.step(noise_pred, t, latents).prev_sample

                # Generate intermediate result every few steps
                if i % 5 == 0 or i == len(timesteps) - 1:
                    # Decode the current latents
                    with torch.no_grad():
                        image = pipeline.vae.decode(latents / pipeline.vae.config.scaling_factor, return_dict=False)[0]
                        image = pipeline.image_processor.postprocess(image, output_type="pil")[0]
                        
                        # Convert PIL Image to numpy array for Gradio
                        yield np.array(image)

        print("Image generation completed.")

    except Exception as e:
        raise Exception(f"Error generating image: {e}")

# Load the model
pipeline_with_lora = load_model_with_lora()