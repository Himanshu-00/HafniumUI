import os
import torch
from model_loader import load_model_with_lora, load_yolo_model
from PIL import Image
import gradio as gr

def process_image(image):
    # Use the YOLO model for face detection
    yolo_model = load_yolo_model()

    # Process the image through YOLO to detect faces
    results = yolo_model(image)
    
    # Assuming the model's output contains bounding boxes, you can overlay them on the image
    annotated_image = results[0].plot()

    # Return the annotated image for display
    return annotated_image

# Set up the Gradio interface
def setup_interface():
    # Load the model with LoRA weights
    model = load_model_with_lora()

    # Gradio Interface
    interface = gr.Interface(
        fn=process_image,  # Function that handles the image input
        inputs=gr.Image(type="pil"),  # Input as an image
        outputs=gr.Image(type="pil")  # Output as an image
    )

    interface.launch()

if __name__ == "__main__":
    setup_interface()
