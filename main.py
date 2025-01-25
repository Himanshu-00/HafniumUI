from gradio_interface import create_gradio_interface
from pipeline import pipeline_with_lora
import webbrowser
import time

def launch_gradio_with_dark_mode(interface):
    # Launch Gradio interface
    interface_url = interface.launch(share=True, debug=True)

    # Ensure the Gradio app is launched, and the URL is retrieved
    time.sleep(2)  # Wait for the app to start

    # The URL for the public interface (Gradio's share URL)
    if isinstance(interface_url, str):
        # Append ?__theme=dark to the Gradio public URL
        dark_mode_url = f"{interface_url}/?__theme=dark"
        print(f"Gradio app with dark mode theme is available at: {dark_mode_url}")

        # Optionally open the URL in the default web browser
        webbrowser.open(dark_mode_url)

if __name__ == "__main__":
    # Create the Gradio interface
    HeliumUI = create_gradio_interface(pipeline_with_lora)

    # Launch Gradio with dark mode
    launch_gradio_with_dark_mode(HeliumUI)
