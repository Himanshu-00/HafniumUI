#gradio_interface.py
import gradio as gr
from pipeline import generate_images
from config import NPROMPT

def create_gradio_interface(pipeline_with_lora):
    theme = gr.themes.Soft(
        primary_hue="indigo", 
        neutral_hue="indigo",
        radius_size="xxl"
    )
    
    with gr.Blocks(theme=theme) as HafniumUI: 
        gr.Markdown("# SDXL with LoRA Integration and Inpainting")

        with gr.Row():
            # Input Column
            with gr.Column():
                input_image = gr.Image(label="Input Image", type="pil")
                color_picker = gr.Radio(
                    choices=[...],  # Keep your existing choices
                    label="Select Professional Suit Color",
                    value="Navy Blue (#000080)",
                    interactive=True
                )
                with gr.Row():
                    guidance_scale = gr.Slider(minimum=1, maximum=20, value=7.5, 
                                             step=0.5, label="Guidance Scale")
                    steps = gr.Slider(minimum=1, maximum=100, value=30,
                                    step=1, label="Number of Steps")

            # Output Column
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
                    num_outputs = gr.Slider(minimum=1, maximum=20, value=1,
                                          step=1, label="Number of Outputs")
                    generate_btn = gr.Button("Generate Image with LoRA", variant="primary")

        # Updated click handler
        generate_btn.click(
            fn=generate_images,
            inputs=[color_picker, guidance_scale, steps, input_image, num_outputs],
            outputs=output_image,
            show_progress=True
        )

    return HafniumUI