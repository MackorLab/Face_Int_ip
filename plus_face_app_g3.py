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
def load_models():
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
    pipe = DiffusionPipeline.from_pretrained("stablediffusionapi/samaritan-3d-cartoon", torch_dtype=torch.float16, scheduler=noise_scheduler, vae=vae, feature_extractor=None, safety_checker=None)
    pipe.to(device)
    
    return pipe, image_encoder_path, ipadapter_sd15_path
pipeline, image_encoder_path, ipadapter_sd15_path = load_models()
def infer(image_path, prompt, negative_prompt, height, width, steps, guide, seed):
    source_image = Image.open(image_path)
    ip_adapter = IPAdapter(pipeline, ipadapter_sd15_path, image_encoder_path, device=device)
    prompt_embeds, negative_prompt_embeds = ip_adapter.get_prompt_embeds(
        source_image,
        prompt=prompt,
        negative_prompt=negative_prompt,
    )
    generator = torch.Generator(device).manual_seed(seed) 
    image = pipeline(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        height=height,
        width=width,
        num_inference_steps=steps,
        guidance_scale=guide,
        generator=generator,
    ).images[0]
    
    return image
inputs = [
    gr.inputs.Image(source="upload", type="filepath", label="Исходный файл - png,jpg"),
    gr.inputs.Textbox(label="Что вы хотите, чтобы ИИ генерировал"),
    gr.inputs.Textbox(label="Что вы не хотите, чтобы ИИ генерировал", default='(deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, wrinkles, old face'),
    gr.Slider(256, 768, 512, step=1, label='Ширина картинки'),
    gr.Slider(256, 768, 512, step=1, label='Высота картинки'),
    gr.Slider(1, 50, value=30, step=1, label='Количество итераций'),
    gr.Slider(2, 15, value=7, label='Шкала расхождения'),
    gr.Slider(label="Зерно", minimum=0, maximum=98765432198, step=1, randomize=True), 
]
output = gr.outputs.Image(type="numpy", label="Result")
iface = gr.Interface(fn=infer, inputs=inputs, outputs=output, title="DIAMONIK7777 - img2img Anime 3D Face Integrator", article="")
iface.launch(debug=True, max_threads=True, share=True, inbrowser=True)
