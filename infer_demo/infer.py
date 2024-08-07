import torch
import os
from diffusers.utils import load_image
from diffusers import EulerDiscreteScheduler
from pipline_StableDiffusion_ConsistentID import ConsistentIDStableDiffusionPipeline

base_model_path = "./stable-diffusion-v1-5"

device = "cuda"
consistentID_path = "./ConsistentID.bin"

### Load base model
pipe = ConsistentIDStableDiffusionPipeline.from_pretrained(
    base_model_path, 
    torch_dtype=torch.float16, 
    use_safetensors=True, 
    variant="fp16"
).to(device)

### Load consistentID_model checkpoint
pipe.load_ConsistentID_model(
    os.path.dirname(consistentID_path),
    subfolder="",
    weight_name=os.path.basename(consistentID_path),
    trigger_word="img",
)     

pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)


### input image
select_images = load_image("./person.jpg")
num_steps = 50
merge_steps = 30
### cinematic photo
prompt = "a woman in a wedding dress"
negative_prompt = "painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured"

generator = torch.Generator(device=device).manual_seed(2024)

images = pipe(
    prompt=prompt,
    width=512,    
    height=512,
    input_id_images=select_images,
    negative_prompt=negative_prompt,
    num_images_per_prompt=1,
    num_inference_steps=num_steps,
    start_merge_step=merge_steps,
    generator=generator,
).images[0]









