import os 
from gradio_interface import create_gradio_interface
from model_loader import download_lora_model


# Ensure model directories exist or create them if they don't
if not os.path.exists("models"):
    os.makedirs("models")
if not os.path.exists("models/loras"):
    os.makedirs("models/loras")
if not os.path.exists("debug_images"):
    os.makedirs("debug_images")

if __name__ == "__main__":
    download_lora_model()
    create_gradio_interface(share=True, debug=True)
