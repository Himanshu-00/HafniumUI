# image_utils.py
from PIL import Image, ImageDraw
import numpy as np
from rembg import remove
from ultralytics import YOLO
from config import CONFIG

# Function to save debug images
def save_debug_image(image, name):
    file_path = os.path.join(CONFIG["debug_dir"], name)
    image.save(file_path)
    print(f"Debug image saved: {file_path}")

# Function to segment image and refine the mask
def segment_and_refine_mask(image, yolo_model):
    segmented = remove(image)  # Segment using rembg (U2Net)
    mask = Image.new("L", image.size, 0)  # Create a blank mask

    # Convert transparency into a grayscale mask
    for y in range(segmented.height):
        for x in range(segmented.width):
            r, g, b, a = segmented.getpixel((x, y))
            mask.putpixel((x, y), a)

    save_debug_image(segmented, "segmented_image.png")
    save_debug_image(mask, "segmented_mask.png")

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

        save_debug_image(image, "yolo_bounding_box.png")
        save_debug_image(inverted_mask, "inverted_refined_mask.png")

        return inverted_mask
    else:
        print("No face detected by YOLO.")
        return mask
