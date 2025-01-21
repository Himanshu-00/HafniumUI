import os
import torch
from PIL import Image, ImageDraw
import numpy as np
from rembg import remove
from ultralytics import YOLO
from config import CONFIG

# Function to save debug images
def save_debug_image(image, filename):
    debug_dir = '/content/HeliumUI/debug_images/'
    print(f"Checking if the directory {debug_dir} exists...")
    
    # Check if the directory exists, if not, create it
    if not os.path.exists(debug_dir):
        print(f"Directory {debug_dir} does not exist. Creating it now.")
        os.makedirs(debug_dir)
    else:
        print(f"Directory {debug_dir} exists.")
    
    # Define the full path for the file
    file_path = os.path.join(debug_dir, filename)
    print(f"Saving image to {file_path}")
    
    # Save the image to the specified path
    try:
        image.save(file_path)
        print(f"Debug image saved to {file_path}")
    except Exception as e:
        print(f"Error while saving the image: {e}")


def generate_image_with_lora(pipeline, prompt, negative_prompt, guidance_scale, steps, input_image, mask):
    # Convert input_image and mask to PIL.Image.Image if they are not already
    if not isinstance(input_image, Image.Image):
        input_image = Image.fromarray(np.array(input_image))

    if not isinstance(mask, Image.Image):
        mask = Image.fromarray(np.array(mask))

    inputs = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "guidance_scale": guidance_scale,
        "num_inference_steps": steps,
        "init_image": input_image,
        "mask_image": mask,
    }

    output = pipeline(**inputs)
    return output


# Function to segment image and refine the mask
def segment_and_refine_mask(image, yolo_model):
    segmented = remove(image)  # Segment using rembg (U2Net)
    mask = Image.new("L", image.size, 0)  # Create a blank mask

    # Convert transparency into a grayscale mask
    for y in range(segmented.height):
        for x in range(segmented.width):
            r, g, b, a = segmented.getpixel((x, y))
            mask.putpixel((x, y), a)

    # Save debug images with correct filenames
    save_debug_image(segmented, "segmented_image.png")
    save_debug_image(mask, "mask_image.png")

    yolo_results = yolo_model(image)  # Perform face detection
    refined_mask = refine_mask_with_bounding_box(image, mask, yolo_results)

    return refined_mask

def refine_mask_with_bounding_box(image, mask, yolo_results):
    if len(yolo_results[0].boxes) > 0:
        box = yolo_results[0].boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        margin_top = 270
        margin_bottom = 30
        margin_left = 60
        margin_right = 60

        y1 = max(0, y1 - margin_top)
        y2 = min(mask.height, y2 + margin_bottom)
        x1 = max(0, x1 - margin_left)
        x2 = min(mask.width, x2 + margin_right)

        mask_array = np.array(mask)
        mask_array[y2:, :] = 0  # Below the bounding box
        mask_array[:y1, :] = 255  # Above the bounding box

        refined_mask = Image.fromarray(mask_array, mode="L")
        inverted_mask = Image.eval(refined_mask, lambda x: 255 - x)

        # Save the debug images with filenames
        save_debug_image(image, "yolo_bounding_box.png")
        save_debug_image(inverted_mask, "inverted_refined_mask.png")

        return inverted_mask
    else:
        print("No face detected by YOLO.")
        return mask
