# #gradio_interface.py
# import gradio as gr
# from pipeline import generate_image_with_lora
# from config import NPROMPT

# def create_gradio_interface(pipeline_with_lora):
#     theme = gr.themes.Soft(
#         primary_hue="indigo", 
#         neutral_hue="indigo",
#         radius_size="xxl"  # Extra large rounded corners
#     )
#     with gr.Blocks(theme=theme) as HafniumUI: 
#         gr.Markdown("# SDXL with LoRA Integration and Inpainting")

#         # Row with two columns
#         with gr.Row():
#             # Left side column with input_image, color selection, guidance_scale, and steps
#             with gr.Column():
#                 input_image = gr.Image(label="Input Image", type="pil")  # Remove tool editor and set type to 'pil'
                
#                 # Add more color options for suit colors with Charcoal as the default
#                 color_picker = gr.Radio(
#                     choices=[
#                         "Charcoal (#3b3b3b)", "Black (#000000)", "Navy Blue (#000080)", 
#                         "Gray (#808080)", "White (#FFFFFF)", "Dark Brown (#654321)", 
#                         "Burgundy (#800020)", "Dark Green (#006400)", "Beige (#F5F5DC)", 
#                         "Light Gray (#D3D3D3)", "Olive Green (#808000)", "Royal Blue (#4169E1)"
#                     ],
#                     label="Select Professional Suit Color",
#                     value="Navy Blue (#000080)",  # Set default value to Charcoal
#                     interactive=True
#                 )
                
#                 # Slider for guidance scale and steps
#                 with gr.Row():
#                     guidance_scale = gr.Slider(minimum=1, maximum=20, value=7.5, step=0.5, label="Guidance Scale", interactive=True)
#                     steps = gr.Slider(minimum=1, maximum=100, value=30, step=1, label="Number of Steps", interactive=True)
                
#             # Right side column for output image
#             with gr.Column():
#                 output_image = gr.Gallery(
#                     label="Generated Images",
#                     elem_id="output_gallery",
#                     columns=3,
#                     preview=True
#                 )

#                 # Add a new slider for the number of images to generate
#                 num_outputs = gr.Slider(minimum=1, maximum=5, value=1, step=1, label="Number of Outputs", interactive=True)

#             # Action for button click
#             generate_btn = gr.Button("Generate Image with LoRA", variant="primary")
            
#             # Button functionality
#             generate_btn.click(
#                 fn=lambda color, gs, steps, img, num_outputs: generate_image_with_lora(
#                     pipeline_with_lora, 
#                     prompt=color, 
#                     negative_prompt=NPROMPT, 
#                     guidance_scale=gs, 
#                     num_steps=steps, 
#                     input_image=img,
#                     num_images=num_outputs
#                 ),
#                 inputs=[color_picker, guidance_scale, steps, input_image, num_outputs],
#                 outputs=output_image
#             )

#     return HafniumUI


# gradio_interface.py
import gradio as gr
from pipeline import generate_image_with_lora

def create_gradio_interface(pipeline_with_lora):
    theme = gr.themes.Soft(
        primary_hue="indigo",
        neutral_hue="slate",
        radius_size="lg",
        font=[gr.themes.GoogleFont("Inter")]
    ).set(
        button_primary_background_fill="*primary_500",
        button_primary_background_fill_hover="*primary_400",
    )
    
    with gr.Blocks(theme=theme, css=".gradio-container {max-width: 900px !important}") as HafniumUI:
        gr.Markdown("""
        # 🖼️ Real-Time Professional Attire Generator
        *Watch your image come to life!*
        """)
        
        with gr.Row(variant="panel"):
            # Left Panel - Inputs
            with gr.Column(scale=1, min_width=300):
                gr.Markdown("## ⚙️ Configuration")
                input_image = gr.Image(label="Upload Reference Photo", type="pil", height=250)
                
                color_picker = gr.Radio(
                    choices=[
                        "Charcoal (#3b3b3b)", "Black (#000000)", "Navy Blue (#000080)", 
                        "Gray (#808080)", "White (#FFFFFF)", "Dark Brown (#654321)", 
                        "Burgundy (#800020)", "Dark Green (#006400)", "Beige (#F5F5DC)", 
                        "Light Gray (#D3D3D3)", "Olive Green (#808000)", "Royal Blue (#4169E1)"
                    ],
                    label="Suit Color Selection",
                    value="Navy Blue (#000080)",
                    interactive=True
                )
                
                with gr.Group():
                    num_outputs = gr.Slider(
                        minimum=1, maximum=5, value=1, step=1,
                        label="Number of Variations",
                        interactive=True
                    )
                    guidance_scale = gr.Slider(
                        minimum=1, maximum=20, value=7.5, step=0.5,
                        label="Creativity Control (Guidance Scale)",
                        interactive=True
                    )
                    steps = gr.Slider(
                        minimum=10, maximum=100, value=30, step=5,
                        label="Generation Steps",
                        interactive=True
                    )
                
                generate_btn = gr.Button(
                    "✨ Generate Variations",
                    variant="primary",
                    size="lg",
                    min_width=200
                )

            # Right Panel - Output
            with gr.Column(scale=2, min_width=600):
                gr.Markdown("## 🎉 Real-Time Generation")
                output_image = gr.Image(
                    label="Live Image Generation",
                    height=600,
                    streaming=True  # Enable streaming updates
                )
        
        # Generation logic with streaming
        def wrapped_generator(*args):
            for result in generate_image_with_lora(pipeline_with_lora, *args):
                yield result  # Stream intermediate images

        generate_btn.click(
            fn=wrapped_generator,
            inputs=[color_picker, guidance_scale, steps, input_image, num_outputs],
            outputs=output_image,
            api_name="generate"
        )

        # UI Enhancements
        gr.Markdown("""
        <div style="text-align: center; padding: 20px; margin-top: 20px; border-radius: 8px; background: var(--block-background-fill);">
            <small>💡 Tip: For best results, use well-lit portrait photos with clear visibility of upper body</small>
        </div>
        """)

    return HafniumUI