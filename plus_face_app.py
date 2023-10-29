import torch
from diffusers import DiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipelineLegacy, DDIMScheduler, AutoencoderKL
from PIL import Image

from ip_adapter.ip_adapter import IPAdapter







vae_model_path = "stabilityai/sd-vae-ft-mse"
image_encoder_path = "/content/Face_Int_ip/models/image_encoder/"
ipadapter_sd15_path = "/content/Face_Int_ip/models/ip-adapter-plus-face_sd15.bin"
device = "cuda"






def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)
vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)






# load SD pipeline
pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, scheduler=noise_scheduler, vae=vae, feature_extractor=None,  safety_checker=None)




image1 = Image.open("assets/O6nQlTcLg9M.jpg")

ip_adapter = IPAdapter(pipe, ipadapter_sd15_path, image_encoder_sd15_path, device=device)

prompt_embeds, negative_prompt_embeds = ip_adapter.get_prompt_embeds(
    image1,
    prompt="Mandala, a close up of a woman wearing a headscarf, arabian beauty, karol bak uhd, very beautiful woman, orange skin. intricate, very beautiful portrait, photo of a beautiful woman, beautiful oriental woman, very extremely beautiful, beautiful arab woman, with beautiful exotic, beautiful portrait, very very beautiful woman, gorgeous woman, gorgeous beautiful woman, beautiful intricate face",
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
image.save("image.png") 

img = load_image("/content/Face_Int_ip/image.png")
img

