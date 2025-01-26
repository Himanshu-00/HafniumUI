from gradio_interface import create_gradio_interface
from pipeline import pipeline_with_lora

if __name__ == "__main__":
    # Initialize the Gradio interface with the pipeline
    HeliumUI = create_gradio_interface(pipeline_with_lora)

    # Launch the Gradio interface with sharing enabled
    launch_output = HeliumUI.launch(share=True, debug=True)

    # Extract the local and public URLs
    local_url, public_url = launch_output[0], launch_output[1]

    # Print URLs with the necessary information
    print("iModel and LoRA weights successfully loaded and moved to device.")
    print(f"* Running on local URL: {local_url}")
    
    # Append `/?__theme=dark` to the public URL if it exists
    if public_url:
        print(f"* Running on public URL: {public_url}/?__theme=dark")
    else:
        print("Public URL not available. Ensure sharing is enabled.")
