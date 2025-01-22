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
            
        if not prompt.strip():
            raise ValueError("Please provide a prompt")

        print(f"Generating image with prompt: '{prompt}', guidance scale: {guidance_scale}, and steps: {num_steps}.")
        
        # Convert input image to PIL and validate
        try:
            input_image = Image.fromarray(input_image).convert("RGB")
            # Resize image to be compatible with SDXL
            input_image = input_image.resize((1024, 1024), Image.LANCZOS)
        except Exception as e:
            raise ValueError(f"Failed to process input image: {e}")

        # Generate mask with error handling
        try:
            mask = segment_and_refine_mask(input_image)
            if mask is None:
                raise ValueError("Failed to generate mask")
            # Ensure mask is same size as input
            mask = mask.resize((1024, 1024), Image.LANCZOS)
        except Exception as e:
            raise ValueError(f"Error in mask generation: {e}")

        # Use the global pipeline
        pipeline = pipeline_with_lora
        if pipeline is None:
            raise ValueError("Pipeline not properly initialized")

        device = pipeline.device
        do_classifier_free_guidance = guidance_scale > 1.0

        # Prepare the mask and image
        mask = np.array(mask)
        if len(mask.shape) == 2:
            mask = mask[:, :, None]
        mask = np.concatenate([mask] * 3, axis=2)
        mask = mask[None].transpose(0, 3, 1, 2)
        mask = torch.from_numpy(mask).to(device=device, dtype=torch.float32) / 255.0

        init_image = np.array(input_image)
        init_image = init_image[None].transpose(0, 3, 1, 2)
        init_image = torch.from_numpy(init_image).to(device=device, dtype=torch.float32) / 127.5 - 1.0

        # Prepare text embeddings
        text_embeddings = pipeline._encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt
        )

        # Prepare latents
        with torch.no_grad():
            # Initialize random noise
            latents = torch.randn(
                (1, pipeline.unet.config.in_channels, 1024 // 8, 1024 // 8),
                device=device,
                dtype=text_embeddings.dtype
            )
            
            # Scale the latents
            latents = latents * pipeline.scheduler.init_noise_sigma

            # Timesteps
            timesteps = pipeline.scheduler.timesteps
            progress(0, desc="Preparing...")

            # Generation loop
            for i, t in enumerate(progress.tqdm(timesteps, desc="Generating")):
                # Expand for classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                
                # Add noise according to the timestep
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

                # Compute previous noisy sample x_t -> x_t-1
                latents = pipeline.scheduler.step(noise_pred, t, latents).prev_sample

                # Generate preview
                if i % 5 == 0 or i == len(timesteps) - 1:
                    with torch.no_grad():
                        image = pipeline.vae.decode(latents / pipeline.vae.config.scaling_factor, return_dict=False)[0]
                        image = pipeline.image_processor.postprocess(image, output_type="pil")[0]
                        yield np.array(image)

        print("Image generation completed successfully.")

    except Exception as e:
        print(f"Error in image generation: {e}")
        # Create error image
        error_image = Image.new('RGB', (512, 512), color='red')
        draw = ImageDraw.Draw(error_image)
        draw.text((10, 10), f"Error: {str(e)}", fill='white')
        yield np.array(error_image)

# Load the model globally
pipeline_with_lora = load_model_with_lora()
