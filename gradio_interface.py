# gradio_interface.py
import gradio as gr
from pipeline import generate_image_with_lora
from config import NPROMPT

def create_gradio_interface(pipeline_with_lora):
    with gr.Blocks() as HafniumUI:
        gr.Markdown("# SDXL with LoRA Integration and Inpainting")

        # Row with two columns
        with gr.Row():
            # Left side column with input_image, color selection, guidance_scale, and steps
            with gr.Column():
                input_image = gr.Image(label="Input Image", tool="editor")
                
                # Add more color options for suit colors with Charcoal as the default
                color_picker = gr.Radio(
                    choices=[
                        "Charcoal (#3b3b3b)", "Black (#000000)", "Navy Blue (#000080)", 
                        "Gray (#808080)", "White (#FFFFFF)", "Dark Brown (#654321)", 
                        "Burgundy (#800020)", "Dark Green (#006400)", "Beige (#F5F5DC)", 
                        "Light Gray (#D3D3D3)", "Olive Green (#808000)", "Royal Blue (#4169E1)"
                    ],
                    label="Select Professional Suit Color",
                    value="Navy Blue (#000080)",  # Set default value to Charcoal
                    interactive=True
                )
                
                # Slider for guidance scale and steps
                with gr.Row():
                    guidance_scale = gr.Slider(minimum=1, maximum=20, value=7.5, step=0.5, label="Guidance Scale")
                    steps = gr.Slider(minimum=1, maximum=100, value=50, step=1, label="Number of Steps")
                
            # Right side column for output image
            with gr.Column():
                output_image = gr.Image(label="Generated Image")
                
                # Add a new slider for the number of images to generate
                num_outputs = gr.Slider(minimum=1, maximum=10, value=1, step=1, label="Number of Outputs")
                # Action for button click
                generate_btn = gr.Button("Generate Image with LoRA")


            generate_btn.click(
                fn=lambda color, gs, steps, img, num_outputs: generate_image_with_lora(
                    pipeline_with_lora, prompt=color, negative_prompt=NPROMPT, guidance_scale=gs, num_steps=steps, input_image=img
                ),
                inputs=[color_picker, guidance_scale, steps, input_image, num_outputs],
                outputs=output_image
            )

    return HafniumUI
