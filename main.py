#main.py
from gradio_interface import create_gradio_interface
from pipeline import pipeline_with_lora

if __name__ == "__main__":
    HeliumUI = create_gradio_interface(pipeline_with_lora)
    # Launch the Gradio interface
    output = HeliumUI.launch(share=True, debug=True)

    # Extract and print the public URL with `/?__theme=dark` appended
    public_url = output[1]  # Gradio returns a tuple (local_url, public_url)
    if public_url:
        print(f"{public_url}/?__theme=dark")
    else:
        print("Public URL not available. Check if sharing is enabled.")
