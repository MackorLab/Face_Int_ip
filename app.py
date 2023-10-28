from diffusers import StableDiffusionPipeline, DDIMScheduler

import torch
from PIL import Image

import config as cfg
from ip_adapter.ip_adapter import IPAdapter

device = "cuda"

pipe = StableDiffusionPipeline.from_single_file("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe.safety_checker = None
pipe.feature_extractor = None
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.to(device)

image1 = Image.open("/assets/input_warrior.jpg")

ip_adapter = IPAdapter(pipe, "ipdapter/model/path", "image/encoder/path", device=device)

prompt_embeds, negative_prompt_embeds = ip_adapter.get_prompt_embeds(
    image1,
    prompt="positive prompt",
    negative_prompt="blurry,",
)

generator = torch.Generator().manual_seed(1)

image = pipe(
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_prompt_embeds,
    num_inference_steps=30,
    guidance_scale=6.0,
    generator=generator,
).images[0]
image.save("image.webp", lossless=True, quality=100)
