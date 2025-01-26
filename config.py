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

A photorealistic portrait of a 23-year-old male model from Canada, exuding professionalism in a tailored navy suit and matching tie.
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

Body Proportion Anomaly Rejection:

Absolute prohibition of unnatural body proportions
Eliminate symmetry disruptions or anatomical inconsistencies
Reject any non-standard limb-to-torso ratios
Prevent skeletal structure distortions


Zero tolerance for:

Disproportionate shoulder widths
Unnaturally elongated or compressed limbs
Irregular torso-to-leg length relationships
Unnatural neck-to-shoulder connections
Biomechanically impossible joint alignments

Visual Composition Constraints:

Absolute prohibition of distracting, complex backgrounds
Total elimination of visual noise and textural complexity
Mandate pristine, monochromatic studio environment
Enforce minimalist neutral backdrop

Technical Image Specifications:

Complete rejection of low-resolution, pixelated imagery
Stringent ultra-high-definition clarity requirements
Prevent digital manipulation artifacts
Enforce naturalistic photographic reproduction

Subject Presentation Parameters:

Suppress exaggerated facial dynamics
Prohibit theatrical makeup and artificial contouring
Mandate organic skin texture and neutral professional appearance

Wardrobe and Styling Directives:

Zero acceptance of informal or inappropriate attire
Complete exclusion of branded elements
Mandate precision-tailored professional vestments

Illumination and Exposure Guidelines:

Reject non-uniform or harsh lighting configurations
Prevent shadow artifacts and lens aberrations
Mandate soft, balanced, naturalistic light distribution

Anatomical Alignment Specifications:

Prohibit perspective-induced bodily distortions
Enforce mathematically accurate human proportionality
Mandate symmetrical, frontally-centered positioning
Ensure exact replication of natural human biomechanics

Professional Accessory Protocols:

Exclusive acceptance of conservative neckwear
Specify matte-finish ties in precise color ranges
Enforce mathematically proportional accessory ratios

Gestural and Positional Refinements:

Eliminate unnatural hand configurations
Mandate organically relaxed, authentic hand placement
Prevent artificially induced postural artifacts

Facial Topography Requirements:

Enforce direct, centered facial orientation
Mandate neutral, professional emotional presentation
Ensure precise eye alignment and authentic dermatological texture""

"No distracting visual elements or cluttered backgrounds with complex patterns, busy textures, or elements that draw attention away from the subject."
"No hands in output"
"No low-resolution, pixelated, blurry, or out-of-focus images; the image must maintain sharp, high-definition details."
"No stylized, cartoon-like, illustrated, painterly, or non-photorealistic rendering styles; the output must appear entirely realistic."
"No exaggerated facial expressions, dramatic makeup, extreme contouring, or overly styled, unnatural-looking hair."
"No informal, wrinkled, ill-fitting, or inappropriate clothing; avoid casual wear, visible logos, flashy designs, or distracting accessories."
"No harsh, uneven, or extreme lighting conditions; avoid overexposure, harsh shadows, lens flares, or dramatic contrasts."
"No environmental context or busy backgrounds; the image must have a simple, controlled studio-like setting."
"No perspective distortions, unflattering camera angles, or unnatural proportions; ensure a direct, centered, and balanced view."
"No heavy post-processing, unnatural color grading, or excessive digital manipulation; maintain a natural and realistic aesthetic."
"No dynamic or candid poses; the subject must maintain a neutral, professional, and controlled posture."
"No facial orientation other than a direct, centered front-facing view with a neutral and professional expression."
"No visible body tilt or unnatural inclinations; ensure upright body alignment without extreme angles (e.g., no 100-degree or diagonal tilt)."
"No unnatural hand positioning or gestures; hands, if visible, must rest naturally without appearing forced or awkward."
"No bold, patterned, reflective, or unconventional neckwear; decorative or alternative styles are to be avoided."
"Maintain precise, anatomically correct body proportions, with natural shoulder width and limb length."
"No visual stretching or compression of body parts; preserve a natural human silhouette."
"Ensure a conservative, solid-color necktie in navy blue or charcoal gray, neatly knotted at the collar, with a matte, non-reflective finish."
"The tie width must be proportional to the suit lapels, positioned centrally for a polished appearance."
"Avoid body positions or angles that distort the natural upright posture; the subject must face the camera with no extreme tilts or unusual rotations."
"No disproportionate body shapes, no stretched or compressed limbs, no irregular shoulder widths, no unnatural silhouettes, no distorted anatomy, no warped proportions."



"""
