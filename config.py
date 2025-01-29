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

A photorealistic portrait of a 23-year-old male model from Canada in a tailored navy suit with a matching tie
He stands confidently against a light gray studio backdrop, illuminated by soft, diffused lighting that highlights the suit's texture and fit. 
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

Body Proportions and Anatomy:
Strict Rejection of Unnatural Body Proportions: Absolutely prohibit any forms of unnatural body proportions. All human figures should follow natural anatomical ratios, with no exaggerated limb lengths, torsos, or necks.
Natural Limb-to-Torso Ratio: Enforce strict adherence to biologically accurate limb-to-torso ratios. No elongation or compression of the arms, legs, or torso beyond realistic human anatomical limits. The limbs must be proportional to the body and torso, maintaining realistic lengths that reflect normal human biomechanics.
Symmetry and Consistency in Proportions: Enforce perfect symmetry in body proportions. The body must appear balanced, with equal proportions on both sides. No asymmetry or deviations from standard human body symmetry.
No Disproportionate Shoulder Widths: Shoulders must be proportionate to the torso with no abnormal widening or narrowing. Avoid excessive shoulder width or disproportionate shoulder-to-waist ratio.
No Disproportionate Limbs: Prevent unnaturally elongated, shortened, or misshaped limbs. Ensure the arms and legs are in proportion to each other and the torso. Eliminate any form of "spaghetti" or "stubby" limbs.
Avoid Distorted Limbs or Joint Misalignment: Eliminate unnatural joint angles or misalignments in elbows, knees, wrists, ankles, and shoulders. All joints must align in a way that reflects natural human anatomy and biomechanics.
No Warped Torso or Leg Lengths: Mandate accurate torso-to-leg length ratios. No elongated torsos or disproportionately short legs. Enforce realistic leg length that aligns with the human body’s natural anatomy.
No Neck Distortions: Ensure the neck is proportionate to the torso. Prevent unnatural neck-to-shoulder connections, such as excessively long or short necks. The neck must be in correct alignment with the body and head.
Prevent Perspective-induced Proportional Distortion: Avoid distortions caused by perspective, camera angles, or viewpoint. The proportions should be accurate regardless of the camera angle or subject's position. The figure must not appear stretched or compressed due to perspective.
No Visible Skin or Skeletal Distortions: Enforce accurate anatomical skeletal and muscular structure. Reject visible or implied distortions of bones, muscles, or skin that result from digital manipulation or unnatural body positions.

Posture, Gesture, and Body Alignment:
Upright and Natural Posture: Ensure that the subject maintains an upright, neutral, and balanced posture at all times. The figure should not exhibit any extreme leaning or slouching, and the body must not tilt unnaturally (e.g., no 100-degree body tilt or sideways body positions).
Symmetrical Body Alignment: Enforce symmetrical alignment of both shoulders, hips, and legs, with the body directly facing the camera. Prevent any form of unnatural torso twisting or unnatural angles. No slouched shoulders or asymmetrical body rotations.
Neutral Hand Positioning: Hands, if visible, must be placed in natural, relaxed positions. Avoid awkward, forced hand gestures or unnatural hand placements (e.g., no hyper-extended fingers, clenched fists, or outstretched palms).
No Distorted or Forced Poses: Ensure the body, limbs, and hands rest in natural, non-stressful positions. Avoid exaggerated poses or extreme stretching. Reject any positions that are biomechanically impossible or forced (e.g., hyper-extended joints, awkward arm positions).
Facial and Head Alignment: Ensure that the head is aligned symmetrically with the torso, and the face is centered and neutral. Reject any tilting of the head or unnatural angular positions (e.g., head tilted excessively to the left or right). The face must always face the camera directly.

Facial Features and Expression:
Neutral Facial Expression: The face must always carry a neutral, professional expression. Avoid exaggerated or unnatural facial dynamics, such as extreme smiles, frowns, or raised eyebrows. Ensure that the mouth and eyes are relaxed and in a neutral state.
No Dramatic Makeup or Contouring: Prohibit the use of theatrical or exaggerated makeup and facial contouring. The skin texture must appear natural, without artificial enhancement or heavy makeup. Facial features should be clear, without excessive cosmetic alterations.
No Stylized, Overly Styled, or Non-Natural Hair: Hair must appear natural and realistic, avoiding overly stylized, slicked-back, or artificial-looking hairstyles. Reject cartoonish, unrealistic hair textures and colors. Hair should look organically styled, with no signs of digital manipulation or distortion.
Eyes Alignment and Expression: The eyes must be aligned with the face symmetrically. No squinting, eye wandering, or extreme eye expressions. Ensure eyes are naturally placed, with no glaring, unnatural sharpness or effects.

Wardrobe and Styling:
Professional and Tailored Clothing Only: Reject casual, wrinkled, ill-fitting, or inappropriate clothing. The attire must be formal, tailored, and professional. Only include suits, shirts, blouses, ties, and formal jackets. Avoid any form of casual wear such as T-shirts, jeans, or shorts.
Exclusion of Brand Logos and Flashy Elements: Prohibit any visible branding, logos, or flashy, distracting elements. Clothing should be simple, clean, and without overt patterns, stripes, or logos that could detract from the overall professional aesthetic.
Neckwear Guidelines: Neckwear, such as ties, should be solid-colored, conservative, and professional. Avoid any neckwear with bold, reflective, or unconventional designs. Ties should be in matte finishes, preferably in dark neutral colors like navy blue or charcoal gray, with proportional width to the suit lapels.
Precise, Proportional Accessory Ratios: Accessories like ties and pocket squares must be of appropriate size, maintaining balance with the attire. Ensure accessories do not overpower or overshadow the overall outfit.

Lighting and Exposure:
Balanced, Soft Lighting: Reject non-uniform or harsh lighting configurations, such as overexposure, strong contrast, or extreme shadows. The lighting should be soft, even, and natural, mimicking a well-balanced studio environment.
No Lens Artifacts or Aberrations: Ensure the image is free from lens flares, glare, or unnatural light artifacts. There should be no visual distractions caused by lighting irregularities.
Realistic, Natural Light Distribution: The light should evenly illuminate the subject without creating harsh highlights or deep shadows. Reject any lighting setups that create unnatural contrasts or overly dramatic lighting effects.

Image Quality and Technical Requirements:
Ultra-High Definition Resolution: Absolutely reject low-resolution, pixelated, blurry, or out-of-focus images. The image must maintain crisp, ultra-high-definition clarity at all times.
No Digital Manipulation Artifacts: Avoid any visible signs of digital manipulation, such as masking, blurring, or unnatural blending. The image must appear as though it was naturally captured in a high-quality photographic setting.
No Over-Processing or Color Grading: Reject any unnatural color grading, heavy post-processing, or excessive adjustments. The image should have a natural, realistic appearance, with no overly saturated colors, extreme contrast, or unnatural filters.

Final Body Specifications:
Accurate Human Proportions: Enforce strict adherence to natural, biologically correct human proportions, with no distortions. The body should retain accurate torso-to-limb ratios, proper shoulder width, and anatomical symmetry.
Natural, Realistic Silhouette: The subject’s silhouette must appear realistic, with no visible stretching or compression of body parts. Ensure that no body parts appear unnaturally large, small, or disproportionate to each other.
Consistent Anatomy Across All Images: For multi-image generation, ensure consistency in anatomical alignment, limb lengths, and body posture across all outputs. There should be no noticeable deviation in body proportions from one image to another.


"No low-resolution, pixelated, blurry, or out-of-focus images; the image must maintain sharp, high-definition details."
"No stylized, cartoon-like, illustrated, painterly, or non-photorealistic rendering styles; the output must appear entirely realistic."
"No exaggerated facial expressions, dramatic makeup, extreme contouring, or overly styled, unnatural-looking hair."
"No informal, wrinkled, ill-fitting, or inappropriate clothing; avoid casual wear, visible logos, flashy designs, or distracting accessories."
"No harsh, uneven, or extreme lighting conditions; avoid overexposure, harsh shadows, lens flares, or dramatic contrasts."
"No busy backgrounds; the image must have a simple, controlled studio-like setting."
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
