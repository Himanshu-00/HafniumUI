from gradio_interface import create_gradio_interface
from pipeline import pipeline_with_lora

if __name__ == "__main__":
    # Create the Gradio interface
    HeliumUI = create_gradio_interface(pipeline_with_lora)
    
    # Launch the interface
    launch_output = HeliumUI.launch(share=True, debug=True)
    
    # Extract local and public URLs from the launch output
    local_url, public_url = launch_output[0], launch_output[1]
    
    # Print the loading message
    print("Model and LoRA weights successfully loaded and moved to device.")
    
    # Print the local URL
    print(f"* Running on local URL:  {local_url}")
    
    # Print the public URL with the theme appended
    if public_url:
        print(f"* Running on public URL: {public_url}/?__theme=dark")
    else:
        print("Public URL not available. Ensure sharing is enabled.")
