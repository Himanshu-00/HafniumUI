#main.py
from gradio_interface import create_gradio_interface
from pipeline import pipeline_with_lora




if __name__ == "__main__":
    HeliumUI = create_gradio_interface(pipeline_with_lora)
    share_link = HeliumUI.launch(share=True, debug=True)

    # Modify the share link to append dark theme
    if share_link and isinstance(share_link, str):
        dark_theme_link = f"{share_link}?__theme=dark"
        print(f"Dark Theme URL: {dark_theme_link}")