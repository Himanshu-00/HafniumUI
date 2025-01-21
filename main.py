import os 
from gradio_interface import create_gradio_interface
from model_loader import download_lora_model



if __name__ == "__main__":
    download_lora_model()
    create_gradio_interface(share=True, debug=True)
