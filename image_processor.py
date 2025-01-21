# image_processor.py

import os
import numpy as np
from PIL import Image, ImageDraw
from rembg import remove  # For segmentation using U2Net
from ultralytics import YOLO
from config import DEBUG_DIR

# Initialize YOLO model
yolo_model = YOLO("yolov8x-face-lindevs.pt")

# Function to save debug images
def save_debug_image(image, name):
    file_path = os.path.join(DEBUG_DIR, name)
    image.save(file_path)
    print(f"Debug image saved: {file_path}")

# Function to segment and refine the mask using rembg and YOLO
def segment_and_refine_mask(image):
    segmented = remove(image)  # Get a transparent image
    mask = Image.new("L", image.size, 0)  # Create a blank mask

    # Convert transparency into a grayscale mask
    for y in range(segmented.height):
        for x in range(segmented.width):
            r, g, b, a = segmented.getpixel((x, y))
            mask.putpixel((x, y), a)

    save_debug_image(segmented, "segmented_image.png")
    save_debug_image(mask, "segmented_mask.png")

    # Perform face detection and refine the mask
    yolo_results = yolo_model(image)
    refined_mask = refine_mask_with_bounding_box(image, mask, yolo_results)

    return refined_mask

# Refine the mask to respect the YOLO bounding box
def refine_mask_with_bounding_box(image, mask, yolo_results):
    if len(yolo_results[0].boxes) > 0:
        box = yolo_results[0].boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        margin_top, margin_bottom, margin_left, margin_right = 270, 30, 60, 60
        y1, y2 = max(0, y1 - margin_top), min(mask.height, y2 + margin_bottom)
        x1, x2 = max(0, x1 - margin_left), min(mask.width, x2 + margin_right)

        mask_array = np.array(mask)
        mask_array[y2:, :] = 0  # Below the bounding box
        mask_array[:y1, :] = 255  # Above the bounding box

        refined_mask = Image.fromarray(mask_array, mode="L")
        inverted_mask = Image.eval(refined_mask, lambda x: 255 - x)

        # Debugging image
        debug_image = image.copy()
        draw = ImageDraw.Draw(debug_image)
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        save_debug_image(debug_image, "yolo_bounding_box.png")
        save_debug_image(inverted_mask, "inverted_refined_mask.png")

        return inverted_mask
    else:
        print("No face detected by YOLO.")
        return mask
