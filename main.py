from gradio_interface import create_gradio_interface
from pipeline import pipeline_with_lora




if __name__ == "__main__":
    HeliumUI = create_gradio_interface(pipeline_with_lora)
    HeliumUI.queue()
    HeliumUI.launch(share=True, debug=True)
