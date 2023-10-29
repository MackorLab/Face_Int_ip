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
image1 = image1.resize((256, 256))



# load ip-adapter
ip_model = IPAdapter(pipe, ipadapter_sd15_path, image_encoder_path, device=device)



images = ip_model.generate(pil_image=image1, num_samples=4, num_inference_steps=50, seed=420,
        prompt="photo of a beautiful girl wearing casual shirt in a garden")
grid = image_grid(images, 1, 4)
grid
