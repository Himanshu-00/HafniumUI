import os
import torch

# Define paths for models
MODEL_PATH = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
LORA_PATH = "models/lora/sd_xl_offset_example-lora_1.0.safetensors"
YOLO_PATH = "yolov8n-face-lindevs.pt"
DEBUG_IMAGE = "debug_images"

# Define URLs for model downloads
LORA_DOWNLOAD_URL = "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_offset_example-lora_1.0.safetensors"

# Configure device preferences here, I'm using MPS for Apple Silicon GPU:
ALLOW_CUDA = True
ALLOW_MPS = False

if torch.cuda.is_available() and ALLOW_CUDA:
    DEVICE = "cuda"
elif torch.backends.mps.is_available() and ALLOW_MPS:
    DEVICE = "mps"
else:
    DEVICE = "cpu"


PROMPT = """

A photorealistic portrait of a 23-year-old male model from Canada, exuding professionalism in a tailored navy suit and a tie.
He stands confidently against a sleek, light gray backdrop inside a modern photo studio, illuminated by soft, diffused lighting that enhances the suit's texture and fit. 
The image is captured in a medium shot, 8K HDR, showcasing the refined elegance of his attire.

"""


PROMPT1 = """

A photorealistic portrait of a 23-year-old male model from Brazil, dressed in a tailored charcoal suit and matching tie that conveys sophistication.
He is positioned against a minimalist black background in a well-lit studio, with dramatic lighting that highlights the contours of his suit and creates a striking visual contrast. 
The image is captured in a three-quarter angle, 8K HDR, emphasizing the suit's craftsmanship.

"""

PROMPT2 = """

A photorealistic portrait of a 23-year-old male model from Germany, presenting a polished look in a tailored light gray suit and matching tie.
He stands against a soft white backdrop in a professional studio setting, enhanced by bright, focused lighting that accentuates the suit's details and creates a clean, modern aesthetic.
The image is captured in a close-up shot, 8K HDR, reflecting the suit's elegance.

"""

PROMPT3 = """

A photorealistic portrait of a 23-year-old male model from Japan, showcasing a tailored black suit and matching tie that radiates confidence. 
He is set against a deep blue background in a contemporary photo studio, with strategic lighting that highlights the suit's sharp lines and luxurious fabric. 
The image is captured in a full-body shot, 8K HDR, illustrating the suit's sophisticated design.

"""

PROMPT4 = """




"""



NPROMPT = """

Visual Composition:

No distracting, complex, or textured backgrounds
Reject cluttered or non-uniform environments
Eliminate busy patterns, decorative elements, or visual noise
Mandate clean, minimalist studio-like setting with neutral backdrop

Image Technical Specifications:

Zero tolerance for low-resolution, pixelated, or blurry imagery
Reject out-of-focus or degraded image quality
Ensure razor-sharp, high-definition details
Prevent excessive digital manipulation or unnatural post-processing
Maintain pure photographic realism without artificial enhancement

Subject Presentation:

No exaggerated facial expressions or dramatic makeup
Eliminate theatrical or artificial contouring
Prohibit stylized, unrealistic hair treatments
Enforce neutral, professional facial composure
Mandate natural skin texture and complexion

Clothing and Styling:

Reject informal, wrinkled, or ill-fitting attire
No visible logos, branded elements, or flashy designs
Eliminate distracting accessories
Require conservative, tailored professional wardrobe
Ensure precise, clean clothing lines and fit

Lighting and Exposure:

Avoid harsh, uneven, or extreme lighting conditions
Prevent overexposure, stark shadows, or dramatic light contrasts
Mandate balanced, natural, soft illumination
Eliminate lens flares, light artifacts, or unnatural highlights

Body Positioning and Proportions:

No perspective distortions or unflattering camera angles
Enforce anatomically precise body proportions
Require direct, centered, perfectly balanced frontal view
Prohibit body tilts, unnatural rotations, or silhouette distortions
Maintain exact shoulder width, limb length, and natural human anatomy

Professional Accessory Guidelines:

Conservative neckwear: solid navy or charcoal gray
Matte, non-reflective tie finish
Precise tie width matching suit lapel proportions
Centrally positioned, perfectly knotted
No decorative or unconventional tie styles

Hand and Gesture Constraints:

Eliminate unnatural hand positioning
Mandate relaxed, organic hand placement
Prevent forced or artificially posed gestures
Ensure hands appear naturally positioned if visible

Facial and Expression Parameters:

Strictly front-facing view
Neutral, professional facial expression
No emotional exaggeration
Precise eye focus and alignment
Natural skin tone and texture""
""" 
