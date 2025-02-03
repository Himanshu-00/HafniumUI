# import os
# from PIL import Image, ImageDraw
# from ultralytics import YOLO
# from rembg import remove  # For segmentation using U2Net
# import numpy as np
# from config import YOLO_PATH, DEBUG_IMAGE


# # Load YOLO face detection model
# yolo_model = YOLO(YOLO_PATH)


# def save_debug_image(image, name):
#     """
#     Saves the given image to the debug directory.
#     """
#     if not os.path.exists("debug_images"):
#         os.makedirs("debug_images")

#     file_path = os.path.join(DEBUG_IMAGE, name)
#     image.save(file_path)
#     print(f"Debug image saved: {file_path}")


# def segment_and_refine_mask(image):
#     """
#     Segments the input image using U2Net (via rembg) and refines the mask with YOLO face detection.
#     """
#     # Step 1: Perform segmentation using rembg
#     segmented = remove(image)  # Get a transparent image
#     mask = Image.new("L", image.size, 0)  # Create a blank mask

#     # Convert transparency into a grayscale mask
#     for y in range(segmented.height):
#         for x in range(segmented.width):
#             r, g, b, a = segmented.getpixel((x, y))
#             mask.putpixel((x, y), a)

#     # Save segmented output
#     save_debug_image(segmented, "segmented_image.png")
#     save_debug_image(mask, "segmented_mask.png")

#     # Step 2: Use YOLO for face detection and refine the mask
#     yolo_results = yolo_model(image)  # Perform face detection with YOLO
#     refined_mask = refine_mask_with_bounding_box(image, mask, yolo_results)

#     return refined_mask


# def refine_mask_with_bounding_box(image, mask, yolo_results):
#     """
#     Refines the mask to respect the YOLO bounding box, sets everything below the bounding box to black (0),
#     and the area above the bounding box to white (255). Then, the mask is inverted. Margins can be added to adjust
#     the bounding box area.
#     """
#     if len(yolo_results[0].boxes) > 0:
#         # Get the first bounding box (assuming one face per image)
#         box = yolo_results[0].boxes.xyxy[0].cpu().numpy()  # Convert tensor to numpy
#         x1, y1, x2, y2 = map(int, box)

#         margin_top = int(0.07 * mask.height)
#         margin_bottom = int(0.01 * mask.height)

#         # Apply margin adjustments
#         y1 = max(0, y1 - margin_top)
#         y2 = min(mask.height, y2 + margin_bottom)

#         # Create the mask
#         mask_array = np.array(mask)

#         # Set everything below the bounding box to black (0)
#         mask_array[y2:, :] = 0

#         # Convert the modified mask array back to an Image object
#         refined_mask = Image.fromarray(mask_array, mode="L")

#         # Invert the mask (black becomes white, and white becomes black)
#         inverted_mask = Image.eval(refined_mask, lambda x: 255 - x)

#         # Draw bounding box on the original image for debugging
#         debug_image = image.copy()
#         draw = ImageDraw.Draw(debug_image)
#         draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

#         # Save debug images
#         save_debug_image(debug_image, "yolo_bounding_box.png")  # Bounding box on original image
#         save_debug_image(inverted_mask, "inverted_refined_mask.png")  # Inverted mask with margins

#         return inverted_mask
#     else:
#         print("No face detected by YOLO.")
#         return mask  # Return the original mask if no face is detected.


# if __name__ == "__main__":
#     input_image_path = "input.png"

#     # Load input image
#     input_image = Image.open(input_image_path).convert("RGBA")

#     # Process the image
#     refined_mask = segment_and_refine_mask(input_image)

#     # Save final refined mask
#     save_debug_image(refined_mask, "final_refined_mask.png")



import gradio as gr
import cv2
import numpy as np
from PIL import Image
from rembg import remove
from ultralytics import YOLO
import os

# Load YOLO model
YOLO_PATH = "yolov8n-face-lindevs.pt"
yolo_model = YOLO(YOLO_PATH)

# Temporary file paths
TEMP_MASK = "temp_mask.png"
TEMP_FINAL_MASK = "final_corrected_mask.png"

# Function to segment and preprocess the mask
def segment_and_refine_mask(image):
    """
    Automatically segment the image and refine it with YOLO.
    """
    # Convert to PIL Image
    image = Image.fromarray(image).convert("RGBA")

    # Step 1: Perform segmentation using rembg
    segmented = remove(image)  # Remove background
    mask = Image.new("L", image.size, 0)  # Create a blank mask

    # Convert transparency into a grayscale mask
    for y in range(segmented.height):
        for x in range(segmented.width):
            r, g, b, a = segmented.getpixel((x, y))
            mask.putpixel((x, y), a)

    # Save initial segmented mask
    mask.save(TEMP_MASK)

    # Step 2: Use YOLO to detect the face and refine the mask
    yolo_results = yolo_model(image)
    
    if len(yolo_results[0].boxes) > 0:
        # Get the first bounding box (assuming one face per image)
        box = yolo_results[0].boxes.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, box)

        # Create mask array
        mask_array = np.array(mask)
        
        # Set everything below the face bounding box to black
        mask_array[y2:, :] = 0
        
        # Convert back to Image object
        refined_mask = Image.fromarray(mask_array, mode="L")

        # Save refined mask
        refined_mask.save(TEMP_MASK)

    return TEMP_MASK

# OpenCV interactive mask editing
def edit_mask_with_opencv():
    """
    Opens OpenCV window to manually edit the mask.
    """
    mask = cv2.imread(TEMP_MASK, cv2.IMREAD_GRAYSCALE)
    drawing = False
    brush_size = 5
    eraser_mode = False  # Toggle between drawing (True) and erasing (False)

    def draw(event, x, y, flags, param):
        nonlocal drawing, eraser_mode
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                color = 255 if not eraser_mode else 0  # White for drawing, Black for erasing
                cv2.circle(mask, (x, y), brush_size, color, -1)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False

    cv2.namedWindow("Edit Mask")
    cv2.setMouseCallback("Edit Mask", draw)

    while True:
        cv2.imshow("Edit Mask", mask)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):  # Press 'q' to save and exit
            break
        elif key == ord("e"):  # Toggle eraser mode
            eraser_mode = not eraser_mode
        elif key == ord("+"):  # Increase brush size
            brush_size += 2
        elif key == ord("-"):  # Decrease brush size
            brush_size = max(1, brush_size - 2)

    # Save corrected mask
    cv2.imwrite(TEMP_FINAL_MASK, mask)
    cv2.destroyAllWindows()

    return TEMP_FINAL_MASK

# Gradio interface
def process_image(image):
    """
    Gradio function to process an image and refine its mask.
    """
    mask_path = segment_and_refine_mask(image)
    return mask_path

def manual_correction():
    """
    Open the OpenCV editor and return the corrected mask.
    """
    corrected_mask = edit_mask_with_opencv()
    return corrected_mask

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# AI-Powered Mask Editor with Manual Correction")

    with gr.Row():
        image_input = gr.Image(type="numpy", label="Upload Image")
        auto_output = gr.Image(label="Automatically Generated Mask")

    process_button = gr.Button("Process Image (Auto Masking)")
    process_button.click(fn=process_image, inputs=image_input, outputs=auto_output)

    gr.Markdown("### Manually Edit Mask (Opens OpenCV Window)")
    correction_button = gr.Button("Manually Edit Mask")
    corrected_output = gr.Image(label="Final Corrected Mask")
    correction_button.click(fn=manual_correction, outputs=corrected_output)

demo.launch()
