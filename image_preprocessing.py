import numpy as np
from PIL import Image, ImageDraw
from rembg import remove
from ultralytics import YOLO
from config import DEBUG_DIR

# Save debug images
def save_debug_image(image, name):
    file_path = os.path.join(DEBUG_DIR, name)
    image.save(file_path)
    print(f"Debug image saved: {file_path}")

# Segment the input image using U2Net and refine with YOLO face detection
def segment_and_refine_mask(image, yolo_model):
    # Step 1: Perform segmentation using rembg
    segmented = remove(image)
    mask = Image.new("L", image.size, 0)

    # Convert transparency into grayscale mask
    for y in range(segmented.height):
        for x in range(segmented.width):
            r, g, b, a = segmented.getpixel((x, y))
            mask.putpixel((x, y), a)

    save_debug_image(segmented, "segmented_image.png")
    save_debug_image(mask, "segmented_mask.png")

    # Step 2: Use YOLO for face detection and refine the mask
    yolo_results = yolo_model(image)
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
        mask_array[y2:, :] = 0
        mask_array[:y1, :] = 255

        refined_mask = Image.fromarray(mask_array, mode="L")
        inverted_mask = Image.eval(refined_mask, lambda x: 255 - x)

        debug_image = image.copy()
        draw = ImageDraw.Draw(debug_image)
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        save_debug_image(debug_image, "yolo_bounding_box.png")
        save_debug_image(inverted_mask, "inverted_refined_mask.png")

        return inverted_mask
    else:
        print("No face detected by YOLO.")
        return mask
