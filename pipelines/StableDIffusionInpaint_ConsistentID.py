from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import cv2
import PIL
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from insightface.app import FaceAnalysis
from safetensors import safe_open
from huggingface_hub.utils import validate_hf_hub_args
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers import StableDiffusionInpaintPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.utils import _get_model_file
from diffusers.utils.torch_utils import is_compiled_module, randn_tensor
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel, ControlNetModel
from functions import process_text_with_markers, masks_for_unique_values, fetch_mask_raw_image, tokenize_and_mask_noun_phrases_ends, prepare_image_token_idx
from functions import ProjPlusModel, masks_for_unique_values
from attention import Consistent_IPAttProcessor, Consistent_AttProcessor, FacialEncoder
from .BaseConsistentID import BaseConsistentIDPipeline
from models.BiSeNet.model import BiSeNet

PipelineImageInput = Union[
    PIL.Image.Image,
    torch.FloatTensor,
    List[PIL.Image.Image],
    List[torch.FloatTensor],
]

class StableDiffusionInpaintConsistentIDPipeline(StableDiffusionInpaintPipeline, BaseConsistentIDPipeline):

    @torch.no_grad()
    # @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: PipelineImageInput = None,
        mask_image: PipelineImageInput = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        strength: float = 1.0,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        original_size: Optional[Tuple[int, int]] = None,
        target_size: Optional[Tuple[int, int]] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        input_id_images: PipelineImageInput = None,
        start_merge_step: int = 0,
        prompt_embeds_text_only: Optional[torch.FloatTensor] = None,
    ):
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)
        
        # 1. Check inputs
        self.check_inputs(
            prompt=prompt,
            height=height,
            width=width,
            strength=strength,
            callback_steps=callback_steps,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )
        if not isinstance(input_id_images, list):
            input_id_images = [input_id_images]
        
        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale >= 1.0
        input_image_file = input_id_images[0]

        faceid_embeds = self.get_prepare_faceid(face_image=input_image_file)
        face_caption = self.get_prepare_llva_caption(input_image_file)
        key_parsing_mask_list, vis_parsing_anno_color = self.get_prepare_facemask(input_image_file)

        assert do_classifier_free_guidance

        # 3. Encode input prompt
        num_id_images = len(input_id_images)

        (
            prompt_text_only,
            clean_input_id,
            key_parsing_mask_list_align,
            facial_token_mask,
            facial_token_idx,
            facial_token_idx_mask,
        ) = self.encode_prompt_with_trigger_word(
            prompt = prompt,
            face_caption = face_caption,
            # prompt_2=None,  
            key_parsing_mask_list=key_parsing_mask_list,
            device=device,
            max_num_facials = 5,
            num_id_images= num_id_images,
            # prompt_embeds= None,
            # pooled_prompt_embeds= None,
            # class_tokens_mask= None,
        )

        # 4. Encode input prompt without the trigger word for delayed conditioning
        encoder_hidden_states = self.text_encoder(clean_input_id.to(device))[0] 

        prompt_embeds = self._encode_prompt(
            prompt_text_only,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt,
        )
        negative_encoder_hidden_states_text_only = prompt_embeds[0:num_images_per_prompt]
        encoder_hidden_states_text_only = prompt_embeds[num_images_per_prompt:]

        # 5. Prepare the input ID images
        prompt_tokens_faceid, uncond_prompt_tokens_faceid = self.get_image_embeds(faceid_embeds, face_image=input_image_file, s_scale=1.0, shortcut=False)

        facial_clip_image, facial_mask = self.get_prepare_clip_image(input_image_file, key_parsing_mask_list_align, image_size=512, max_num_facials=5)
        facial_clip_images = facial_clip_image.unsqueeze(0).to(device, dtype=self.torch_dtype)
        facial_token_mask = facial_token_mask.to(device)
        facial_token_idx_mask = facial_token_idx_mask.to(device)
        negative_encoder_hidden_states = negative_encoder_hidden_states_text_only

        cross_attention_kwargs = {}

        # 6. Get the update text embedding
        prompt_embeds_facial, uncond_prompt_embeds_facial = self.get_facial_embeds(encoder_hidden_states, negative_encoder_hidden_states, \
                                                            facial_clip_images, facial_token_mask, facial_token_idx_mask)

        prompt_embeds = torch.cat([prompt_embeds_facial, prompt_tokens_faceid], dim=1)
        negative_prompt_embeds = torch.cat([uncond_prompt_embeds_facial, uncond_prompt_tokens_faceid], dim=1)

        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )        
        prompt_embeds_text_only = torch.cat([encoder_hidden_states_text_only, prompt_tokens_faceid], dim=1)
        prompt_embeds = torch.cat([prompt_embeds, prompt_embeds_text_only], dim=0)

        # 7. Prepare images
        init_image = self.image_processor.preprocess(
            input_image_file, height=height, width=width,  # resize_mode='default', crops_coords=None
        )
        init_image = init_image.to(dtype=torch.float32)

        mask = self.mask_processor.preprocess(
            mask_image, height=height, width=width, # resize_mode='default', crops_coords=None
        )

        masked_image = init_image * (mask < 0.5)
        _, _, height, width = init_image.shape

        # 8. Pprepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self.get_timesteps(
            num_inference_steps=num_inference_steps, strength=strength, device=device
        )
        # at which timestep to set the initial noise (n.b. 50% if strength is 0.5)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
        # create a boolean to check if the strength is set to 1. if so then initialise the latents with pure noise
        is_strength_max = strength == 1.0
        self._num_timesteps = len(timesteps)

        # 9.Prepare latent variables
        num_channels_latents = self.vae.config.latent_channels
        num_channels_unet = self.unet.config.in_channels
        return_image_latents = num_channels_unet == 4
        latents_outputs = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
            image=init_image,
            timestep=latent_timestep,
            is_strength_max=is_strength_max,
            return_noise=True,
            return_image_latents=return_image_latents,
        )

        if return_image_latents:
            latents, noise, image_latents = latents_outputs
        else:
            latents, noise = latents_outputs

        # 10. Prepare mask latent variables
        mask, masked_image_latents = self.prepare_mask_latents(
            mask,
            masked_image,
            batch_size * num_images_per_prompt,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            do_classifier_free_guidance,
        )

        # 11. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        
        (
            null_prompt_embeds,
            augmented_prompt_embeds,
            text_prompt_embeds,
        ) = prompt_embeds.chunk(3)

        # 12. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                if i <= start_merge_step:
                    current_prompt_embeds = torch.cat(
                        [null_prompt_embeds, text_prompt_embeds], dim=0
                    )
                else:
                    current_prompt_embeds = torch.cat(
                        [null_prompt_embeds, augmented_prompt_embeds], dim=0
                    )
                
                # predict the noise residual
                if num_channels_unet == 9:
                    latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)

                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=current_prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                else:
                    assert 0, 'Not Implemented'
                
                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                if num_channels_unet == 4:
                    init_latents_proper = image_latents
                    if do_classifier_free_guidance:
                        init_mask, _ = mask.chunk(2)
                    else:
                        init_mask = mask

                    if i < len(timesteps) - 1:
                        noise_timestep = timesteps[i + 1]
                        init_latents_proper = self.scheduler.add_noise(
                            init_latents_proper, noise, torch.tensor([noise_timestep])
                        )
                    latents = (1 - init_mask) * init_latents_proper + init_mask * latents
                
                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        # If we do sequential model offloading, let's offload unet and controlnet
        # manually for max memory savings
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.unet.to("cpu")
            torch.cuda.empty_cache()

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[
                0
            ]
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
