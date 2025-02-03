#gradio_interface.py
import gradio as gr
from pipeline import generate_images
from config import NPROMPT

def create_gradio_interface(pipeline_with_lora):
    theme = gr.themes.Soft(
        primary_hue="indigo", 
        neutral_hue="indigo",
        radius_size="xxl"  # Extra large rounded corners
    )
    with gr.Blocks(theme=theme) as HafniumUI: 

        gr.Markdown("# HafniumUI")

        # Row with two columns
        with gr.Row():
            # Left side column with input_image, color selection, guidance_scale, and steps
            with gr.Column():
                input_image = gr.Image(label="Input Image", tool="editor", type="pil")  # Remove tool editor and set type to 'pil'
                
                # Add more color options for suit colors with Charcoal as the default
                color_picker = gr.Radio(
                    choices=[
                        "Charcoal (#3b3b3b)", "Black (#000000)", "Navy Blue (#000080)", 
                        "Gray (#808080)", "White (#FFFFFF)", "Dark Brown (#654321)", 
                        "Burgundy (#800020)", "Dark Green (#006400)", "Beige (#F5F5DC)", 
                        "Light Gray (#D3D3D3)", "Olive Green (#808000)", "Royal Blue (#4169E1)"
                    ],
                    label="Select Professional Suit Color",
                    value="Navy Blue (#000080)",  # Set default value to Blue
                    interactive=True
                )
                
                # Slider for guidance scale and steps
                with gr.Row():
                    guidance_scale = gr.Slider(minimum=1, maximum=20, value=7.5, step=0.5, label="Guidance Scale", interactive=True)
                    steps = gr.Slider(minimum=1, maximum=100, value=30, step=1, label="Number of Steps", interactive=True)
                
            # Right side column for output image
            with gr.Column(min_width=800):
                output_image = gr.Gallery(
                    label="Generated Images",
                    elem_id="output_gallery",
                    columns=4,
                    preview=True,
                    object_fit="contain",
                    height=600
                )

                with gr.Row():
                    # Add a new slider for the number of images to generate
                    num_outputs = gr.Slider(minimum=1, maximum=20, value=1, step=1, label="Number of Outputs", interactive=True)
                    # Action for button click
                    generate_btn = gr.Button("Generate Image with LoRA", variant="primary")
                
                
            # Button functionality
            generate_btn.click(
                fn=generate_images,
                inputs=[color_picker, guidance_scale, steps, input_image, num_outputs],
                outputs=output_image,
                show_progress=True
            )


    return HafniumUI
