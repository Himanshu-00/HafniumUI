from gradio_interface import create_gradio_interface
from pipeline import pipeline_with_lora

if __name__ == "__main__":
    # Create the Gradio interface
    HeliumUI = create_gradio_interface(pipeline_with_lora)
    
    # Launch the Gradio app and capture the URLs
    print("Model and LoRA weights successfully loaded and moved to device.")
    try:
        local_url, public_url = HeliumUI.launch(share=True, debug=True)
        
        # Print the URLs with the dark theme appended to the public URL
        print(f"* Running on local URL:  {local_url}")
        if public_url:
            print(f"* Running on public URL: {public_url}/?__theme=dark")
        else:
            print("Public URL not available. Ensure sharing is enabled.")
    except Exception as e:
        print(f"An error occurred while launching the Gradio app: {e}")
