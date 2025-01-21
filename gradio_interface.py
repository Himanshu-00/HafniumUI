# gradio_interface.py
import gradio as gr
from pipeline import generate_image_with_lora

def create_gradio_interface(pipeline_with_lora):
    with gr.Blocks() as HeliumUI:
        gr.Markdown("# SDXL with LoRA Integration and Inpainting")
        
        # Row with two columns
        with gr.Row():
            # Left side column
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here...")
                negative_prompt = gr.Textbox(label="Negative Prompt", placeholder="What you don't want in the image...")
                with gr.Row():
                    guidance_scale = gr.Slider(minimum=1, maximum=20, value=7.5, step=0.5, label="Guidance Scale")
                    steps = gr.Slider(minimum=1, maximum=100, value=30, step=1, label="Number of Steps")
                input_image = gr.Image(label="Input Image", tool="editor")
                generate_btn = gr.Button("Generate Image with LoRA")
                
                # Add progress indicator
                progress = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=0,
                    step=1,
                    label="Generation Progress",
                    interactive=False
                )

            # Right side column
            with gr.Column():
                output_image = gr.Image(label="Generated Image")

        def generate_with_progress(prompt, neg_prompt, gs, steps, img):
            # Initialize progress state
            progress_value = 0
            
            def update_progress(preview_image, progress_fraction):
                nonlocal progress_value
                progress_value = progress_fraction * 100
                return preview_image, progress_value
            
            return generate_image_with_lora(
                pipeline_with_lora,
                prompt,
                neg_prompt,
                gs,
                steps,
                img,
                progress=update_progress
            )

        # Action for button click
        generate_btn.click(
            fn=generate_with_progress,
            inputs=[prompt, negative_prompt, guidance_scale, steps, input_image],
            outputs=[output_image, progress]
        )

    return HeliumUI