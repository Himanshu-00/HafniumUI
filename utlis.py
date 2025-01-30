from pipeline import generate_images
from config import PROMPT


#Prompt Handler
def prompt_handler(gender, age, color_picker, color, guidance_scale, steps, input_image, num_outputs):
    # Replace placeholders in prompt
    prompt = PROMPT.replace("[Age]", str(age)) \
                         .replace("[Gender]", gender) \
                    
    return generate_images(prompt, color_picker, guidance_scale, steps, input_image, num_outputs, color)
