# gradio_interface.py

import gradio as gr
from image_generator import generate_image_with_lora

def create_gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# SDXL with LoRA Integration and Inpainting")

        with gr.Row():
            # Left column
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here...")
                negative_prompt = gr.Textbox(label="Negative Prompt", placeholder="What you don't want in the image...")
                with gr.Row():
                    guidance_scale = gr.Slider(minimum=1, maximum=20, value=7.5, step=0.5, label="Guidance Scale")
                    steps = gr.Slider(minimum=1, maximum=100, value=30, step=1, label="Number of Steps")
                input_image = gr.Image(label="Input Image", tool="editor")
                generate_btn = gr.Button("Generate Image with LoRA")

            # Right column
            with gr.Column():
                output_image = gr.Image(label="Generated Image")

        # Action for button click
        generate_btn.click(
            fn=lambda prompt, neg_prompt, gs, steps, img: generate_image_with_lora(
                prompt, neg_prompt, gs, steps, img
            ),
            inputs=[prompt, negative_prompt, guidance_scale, steps, input_image],
            outputs=output_image
        )

    demo.launch(share=True, debug=True)
