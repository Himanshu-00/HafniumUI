import gradio as gr
from model_loader import load_model_with_lora, load_yolo_model
from image_preprocessing import segment_and_refine_mask
from config import DEBUG_DIR

def gradio_interface():
    pipeline_with_lora = load_model_with_lora()
    yolo_model = load_yolo_model()

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
                pipeline_with_lora, prompt, neg_prompt, gs, steps, img, yolo_model
            ),
            inputs=[prompt, negative_prompt, guidance_scale, steps, input_image],
            outputs=output_image
        )

    demo.launch(share=True, debug=True)
