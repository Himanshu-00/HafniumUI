import os
from PIL import Image, ImageDraw
from ultralytics import YOLO
from rembg import remove  # For segmentation using U2Net
import numpy as np
from config import YOLO_PATH, DEBUG_IMAGE


# Load YOLO face detection model
yolo_model = YOLO(YOLO_PATH)


def save_debug_image(image, name):
    """
    Saves the given image to the debug directory.
    """
    if not os.path.exists("debug_images"):
        os.makedirs("debug_images")

    file_path = os.path.join(DEBUG_IMAGE, name)
    image.save(file_path)
    print(f"Debug image saved: {file_path}")

def segment_and_refine_mask(image):
    """
    Segments the input image using U2Net (via rembg) and refines the mask with YOLO face detection.
    """
    # Step 1: Perform segmentation using rembg
    segmented = remove(image)  # Get a transparent image
    mask = Image.new("L", image.size, 0)  # Create a blank mask

    # Convert transparency into a grayscale mask
    for y in range(segmented.height):
        for x in range(segmented.width):
            r, g, b, a = segmented.getpixel((x, y))
            mask.putpixel((x, y), a)

    # Save segmented output
    save_debug_image(segmented, "segmented_image.png")
    save_debug_image(mask, "segmented_mask.png")

    # Step 2: Use YOLO for face detection and refine the mask
    yolo_results = yolo_model(image)  # Perform face detection with YOLO
    refined_mask = refine_mask_with_bounding_box(image, mask, yolo_results)

    return refined_mask

def refine_mask_with_bounding_box(image, mask, yolo_results):
    """
    Refines the mask to respect the YOLO bounding box, sets everything below the bounding box to black (0),
    and the area above the bounding box to white (255). Then, the mask is inverted. Margins can be added to adjust
    the bounding box area.
    """
    if len(yolo_results[0].boxes) > 0:
        # Get the first bounding box (assuming one face per image)
        box = yolo_results[0].boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # margin_top = int(0.05 * mask.height)
        # margin_bottom = int(0.01 * mask.height)

        # # Apply margin adjustments
        # y1 = max(0, y1 - margin_top)
        # y2 = min(mask.height, y2 + margin_bottom)

        # Tighten vertical bounds, focusing strictly on face
        face_height = y2 - y1
        y1 += int(face_height * 0.1)  # Shift top down slightly
        y2 -= int(face_height * 0.3)  # Crop bottom more aggressively



        # Create the mask 
        mask_array = np.array(mask)

        # Set everything below the bounding box to black (0)
        mask_array[y2:, :] = 0 

        # Convert the modified mask array back to an Image object
        refined_mask = Image.fromarray(mask_array, mode="L")

        # Invert the mask (black becomes white, and white becomes black)
        inverted_mask = Image.eval(refined_mask, lambda x: 255 - x)

        # Draw bounding box on the original image for debugging
        debug_image = image.copy()
        draw = ImageDraw.Draw(debug_image)
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        # Save debug images
        save_debug_image(debug_image, "yolo_bounding_box.png")  # Bounding box on original image
        save_debug_image(inverted_mask, "inverted_refined_mask.png")  # Inverted mask with margins

        return inverted_mask
    else:
        print("No face detected by YOLO.")
        return mask  # Return the original mask if no face is detected.


