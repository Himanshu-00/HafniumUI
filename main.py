#main.py
from gradio_interface import create_gradio_interface
from pipeline import pipeline_with_lora, generate_images
from config import PROMPT


#Prompt Handler
def prompt_handler(gender, age, color_picker, guidance_scale, steps, input_image, num_outputs):
    # Replace placeholders in prompt
    prompt = PROMPT.replace("[Age]", str(age)) \
                         .replace("[Gender]", gender) \
                    
    return generate_images(prompt, color_picker, guidance_scale, steps, input_image, num_outputs)


if __name__ == "__main__":
    HeliumUI = create_gradio_interface(pipeline_with_lora)
    HeliumUI.launch(share=True, debug=True)
