import gradio as gr
import torch
from diffusers import DiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipelineLegacy, DDIMScheduler, AutoencoderKL
from PIL import Image
from ip_adapter.ip_adapter import IPAdapter
device = "cuda" if torch.cuda.is_available() else "cpu"
def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid
def infer(image_path, prompt, negative_prompt):
    vae_model_path = "stabilityai/sd-vae-ft-mse"
    image_encoder_path = "/content/Face_Int_ip/models/image_encoder/"
    ipadapter_sd15_path = "/content/Face_Int_ip/models/ip-adapter-plus-face_sd15.bin"
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
    pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, scheduler=noise_scheduler, vae=vae, feature_extractor=None, safety_checker=None)
    pipe.to(device)
    source_image = Image.open(image_path)
    ip_adapter = IPAdapter(pipe, ipadapter_sd15_path, image_encoder_path, device=device)
    prompt_embeds, negative_prompt_embeds = ip_adapter.get_prompt_embeds(
        source_image,
        prompt=prompt,
        negative_prompt=negative_prompt,
    )
    generator = torch.Generator().manual_seed(1)
    image = pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        num_inference_steps=30,
        guidance_scale=6.0,
        generator=generator,
    ).images[0]
    
    return image
inputs = [
    gr.Image(source="upload", type="filepath", label="Raw Image. Must Be .png"),
    gr.inputs.Textbox(label="Prompt"),
    gr.inputs.Textbox(label="Negative Prompt"),
]
output = gr.outputs.Image(type="filepath", label="Result")
iface = gr.Interface(fn=infer, inputs=inputs, outputs=output, title="Stable Diffusion", article="")
iface.launch(debug=True, max_threads=True, share=True, inbrowser=True)
