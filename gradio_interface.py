# gradio_interface.py
import gradio as gr
from config import CONFIG
from PIL import Image
from model import load_model_with_lora
from image_utils import generate_image_with_lora, segment_and_refine_mask
from ultralytics import YOLO

# Load YOLO face detection model
yolo_model = YOLO(CONFIG["yolo_model_path"])

# Load the model with LoRA
pipeline_with_lora = load_model_with_lora()

# Function to generate image with LoRA
def generate_image(prompt, negative_prompt, guidance_scale, steps, input_image):
    input_image = Image.fromarray(input_image).convert("RGB")
    mask = segment_and_refine_mask(input_image, yolo_model)
    image = generate_image_with_lora(pipeline_with_lora, prompt, negative_prompt, guidance_scale, steps, input_image, mask)
    return image

# Create Gradio interface
with gr.Blocks() as Helium:
    gr.Markdown("# SDXL with LoRA Integration and Inpainting")

    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here...")
            negative_prompt = gr.Textbox(label="Negative Prompt", placeholder="What you don't want in the image...")
            with gr.Row():
                guidance_scale = gr.Slider(minimum=1, maximum=20, value=CONFIG["guidance_scale"], step=0.5, label="Guidance Scale")
                steps = gr.Slider(minimum=1, maximum=100, value=CONFIG["steps"], step=1, label="Number of Steps")
            input_image = gr.Image(label="Input Image", tool="editor")
            generate_btn = gr.Button("Generate Image with LoRA")

        with gr.Column():
            output_image = gr.Image(label="Generated Image")

    generate_btn.click(
        fn=generate_image,
        inputs=[prompt, negative_prompt, guidance_scale, steps, input_image],
        outputs=output_image
    )

Helium.launch(share=True, debug=True)
