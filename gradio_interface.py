# gradio_interface.py
import gradio as gr
from model_loader import load_model_with_lora
from image_preprocessing import segment_and_refine_mask
from config import DEBUG_DIR

def generate_image_with_lora(pipeline, prompt, negative_prompt, guidance_scale, num_steps, input_image):
    try:
        if not prompt.strip():
            raise Exception("Please provide a prompt.")
        print(f"Generating image with prompt: '{prompt}', negative prompt: '{negative_prompt}', guidance scale: {guidance_scale}, and steps: {num_steps}.")
        input_image = Image.fromarray(input_image).convert("RGB")
        mask = segment_and_refine_mask(input_image)

        with torch.no_grad():
            image = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_steps,
                image=input_image,
                mask_image=mask
            ).images[0]

        print("Image generated successfully.")
        return image

    except Exception as e:
        raise Exception(f"Error generating image: {e}")

def create_gradio_interface(pipeline):
    with gr.Blocks() as demo:
        gr.Markdown("# SDXL with LoRA Integration and Inpainting")

        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here...")
                negative_prompt = gr.Textbox(label="Negative Prompt", placeholder="What you don't want in the image...")
                with gr.Row():
                    guidance_scale = gr.Slider(minimum=1, maximum=20, value=7.5, step=0.5, label="Guidance Scale")
                    steps = gr.Slider(minimum=1, maximum=100, value=30, step=1, label="Number of Steps")
                input_image = gr.Image(label="Input Image", tool="editor")
                generate_btn = gr.Button("Generate Image with LoRA")

            with gr.Column():
                output_image = gr.Image(label="Generated Image")

        generate_btn.click(
            fn=lambda prompt, neg_prompt, gs, steps, img: generate_image_with_lora(
                pipeline, prompt, neg_prompt, gs, steps, img
            ),
            inputs=[prompt, negative_prompt, guidance_scale, steps, input_image],
            outputs=output_image
        )

    return demo
