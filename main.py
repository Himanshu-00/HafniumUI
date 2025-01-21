# main.py
import os
from model_loader import load_model_with_lora
from config import MODEL_DIR, LOLA_DIR, DEBUG_DIR
from gradio_interface import create_gradio_interface

def create_directories():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    if not os.path.exists(LOLA_DIR):
        os.makedirs(LOLA_DIR)
    if not os.path.exists(DEBUG_DIR):
        os.makedirs(DEBUG_DIR)

def main():
    create_directories()

    pipeline_with_lora = load_model_with_lora()

    # Start Gradio interface
    demo = create_gradio_interface(pipeline_with_lora)
    demo.launch(share=True, debug=True)

if __name__ == "__main__":
    main()
