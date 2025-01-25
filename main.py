from gradio_interface import create_gradio_interface
from pipeline import pipeline_with_lora
import webbrowser
import time

def launch_gradio_with_dark_mode(interface):
    # Launch Gradio interface and capture the URLs (local and public)
    local_url, public_url = interface.launch(share=True, debug=True)

    # Append ?__theme=dark to the public URL
    dark_mode_url = f"{public_url}/?__theme=dark"
    print(f"Gradio app with dark mode theme is available at: {dark_mode_url}")

    # Optionally open the URL in the default web browser
    webbrowser.open(dark_mode_url)

if __name__ == "__main__":
    # Create the Gradio interface
    HeliumUI = create_gradio_interface(pipeline_with_lora)

    # Launch Gradio with dark mode
    launch_gradio_with_dark_mode(HeliumUI)
