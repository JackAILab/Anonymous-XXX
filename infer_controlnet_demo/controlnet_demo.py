# Import necessary libraries
import torch
import os
from diffusers import ControlNetModel, DDIMScheduler
from pipelines.StableDIffusionControlNet_ConsistentID import StableDIffusionControlNetConsistentIDPipeline
from diffusers.utils import load_image
import numpy as np
import cv2
from PIL import Image
from controlnet_aux import OpenposeDetector
import pdb

# Set device and define model paths
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("GPU is available. Using GPU:", torch.cuda.get_device_name(device))
else:
    device = torch.device('cpu')
    print("GPU is not available. Using CPU.")

base_model_path = "./Realistic_Vision_V6.0_B1_noVAE"
consistentID_path = "./ConsistentID-v1.bin"

# Load initial and mask images
init_image_url = "./scarlett_johansson.jpg"
ref_image_url = "./albert_einstein.jpg"
init_image = load_image(init_image_url)
ref_image = load_image(ref_image_url)

# Resize images
init_image = init_image.resize((512, 512))
ref_image = ref_image.resize((512, 512))

# Create control image using Canny edge detection
def make_canny_condition(image):
    image = np.array(image)
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    image = Image.fromarray(image)
    return image

# pdb.set_trace()

### canny-control
# control_image = make_canny_condition(ref_image)

### openpose-control
### include_body=True, include_hand=False, include_face=False, hand_and_face=None,
processor = OpenposeDetector.from_pretrained(pretrained_model_or_path="/data2/huangjiehui_m22/pretrained_model/ControlNet", filename='/data2/huangjiehui_m22/pretrained_model/ControlNet/annotator/ckpts/body_pose_model.pth')
control_image = processor(ref_image, include_face=True)

# pdb.set_trace()
# Load control model for inpainting
controlnet = ControlNetModel.from_pretrained(
    "./control_v11p_sd15_openpose",
    torch_dtype=torch.float16,
).to(device) 

# Load base model
pipe = StableDIffusionControlNetConsistentIDPipeline.from_pretrained(
    base_model_path, 
    controlnet=controlnet, 
    torch_dtype=torch.float16, 
    use_safetensors=True, 
    variant="fp16",
).to(device)

# Load ConsistentID model checkpoint
pipe.load_ConsistentID_model(
    os.path.dirname(consistentID_path),
    subfolder="",
    weight_name=os.path.basename(consistentID_path),
    trigger_word="img",
)

# Set up scheduler
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
pipe.to(device)

# Set generator with seed
generator = torch.Generator(device=device).manual_seed(2024)

# Hyper-parameters
num_steps = 50
merge_steps = 30

# Define prompt and parameters
prompt = "cinematic photo, A woman, in a forest, adventuring, 50mm photograph, half-length portrait, film, bokeh, professional, 4k, highly detailed"
negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality, blurry, ((((ugly)))), (((duplicate))), ((morbid)), ((mutilated)), [out of frame], extra fingers, mutated hands, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), (((deformed))), ((ugly)), blurry, ((bad anatomy)), (((bad proportions))), ((extra limbs)), cloned face, (((disfigured))). out of frame, ugly, extra limbs, (bad anatomy), gross proportions, (malformed limbs), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), mutated hands, (fused fingers), (too many fingers), (((long neck)))"

# Generate the image
images = pipe(
    prompt=prompt,
    width=512,    
    height=512,
    # controlnet_conditioning_scale=0.5,
    # mask_image=mask_image,
    control_image=control_image,
    input_id_images=init_image,
    negative_prompt=negative_prompt,
    num_images_per_prompt=1,
    num_inference_steps=num_steps,
    start_merge_step=merge_steps,
    generator=generator,
).images[0]

# Save the resulting image
image_save_path = "./result.jpg"
images.save(image_save_path)
print(f"saved at: {image_save_path}")




















