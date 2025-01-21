import os
from model_loader import download_lora_model
from gradio_interface import demo

# Ensure the model and LoRA weights are downloaded and available
download_lora_model()

# Launch the Gradio interface
if __name__ == "__main__":
    demo.launch(share=True, debug=True)
