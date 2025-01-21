import os 
from gradio_interface import create_gradio_interface, pipeline_with_lora





if __name__ == "__main__":
    interface = create_gradio_interface(pipeline_with_lora)
    interface.launch(share=True, debug=True)
