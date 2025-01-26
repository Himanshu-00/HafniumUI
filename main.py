# main.py
from gradio_interface import create_gradio_interface
from pipeline import pipeline_with_lora

if __name__ == "__main__":
    HeliumUI = create_gradio_interface(pipeline_with_lora)
    public_url = HeliumUI.launch(share=True, debug=True)
    
    # Print dark theme URL
    if public_url:
        dark_theme_url = f"{public_url}?__theme=dark"
        print(f"Dark Theme URL: {dark_theme_url}")