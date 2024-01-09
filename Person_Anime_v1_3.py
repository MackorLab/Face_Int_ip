from diffusers import (
    StableDiffusionControlNetInpaintPipeline,
    ControlNetModel,
    LCMScheduler,
)
import torch
from diffusers.utils import load_image
from PIL import Image
import cv2
import numpy as np

def get_canny_filter(image):
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    return canny_image

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_canny",
    torch_dtype=torch.float16,
)

pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "Yntec/3Danimation",
    controlnet=controlnet,
    torch_dtype=torch.float16,
    requires_safety_checker=False,
    safety_checker=None,
)


device = "cuda"

pipe.to(device)

# Load ip-adapter
pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
