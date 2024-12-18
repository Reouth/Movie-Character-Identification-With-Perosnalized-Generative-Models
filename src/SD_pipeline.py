from PIL import Image
import torch
from torch.nn import functional as F
from typing import List, Optional, Union

from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers import DiffusionPipeline

from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import numpy as np
from torchvision import transforms

# Helper class to track average loss
class AverageMeter:
    def __init__(self, name=None):
        self.name = name
        self.reset()

    def reset(self):
        self.sum = self.count = self.avg = 0

    def update(self, val, n=1):
        """
        Update the average with a new value.

        Args:
            val (float): The new value to include in the average.
            n (int): The weight of the new value (default is 1).
        """
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# Custom implementation of the Stable Diffusion pipeline
class StableDiffusionPipeline(DiffusionPipeline):
    r"""
    Pipeline for image editing and image identification using Stable Diffusion.
    Inherits from `DiffusionPipeline`.

    Attributes:
        vae: Variational Auto-Encoder (VAE) for encoding/decoding images to/from latent space.
        text_encoder: CLIP text encoder for embedding textual prompts.
        tokenizer: Tokenizer for converting text to tokens.
        unet: Conditional U-Net for denoising latent representations.
        scheduler: Scheduler for managing denoising steps.
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[DDIMScheduler, LMSDiscreteScheduler]
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler
        )

    def enable_attention_slicing(self, slice_size: Optional[Union[str, int]] = "auto"):
        """
        Enable sliced attention computation to save memory.

        Args:
            slice_size: Number of slices for attention computation ("auto" defaults to halving the head size).
        """
        if slice_size == "auto":
            slice_size = self.unet.config.attention_head_dim // 2
        self.unet.set_attention_slice(slice_size)

    def disable_attention_slicing(self):
        """
        Disable sliced attention computation.
        """
        self.enable_attention_slicing(None)

    @torch.no_grad()
    def generateImage(self,
        cond_embeddings,
        seed: int = 0,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: Optional[int] = 50,
        output_type: Optional[str] = "pil",
        guidance_scale: float = 7.5
    ):
        """
        Generate an image from conditional embeddings using diffusion.

        Args:
            cond_embeddings: Embeddings guiding image generation.
            seed: Random seed for reproducibility.
            height: Height of the generated image (must be divisible by 8).
            width: Width of the generated image (must be divisible by 8).
            num_inference_steps: Number of diffusion steps.
            output_type: Format of the output image ("pil" or "numpy").
            guidance_scale: Guidance strength for classifier-free guidance.

        Returns:
            Generated images in the specified format.
        """
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` must be divisible by 8, but received {height} and {width}.")

        # Determine if classifier-free guidance is used
        do_classifier_free_guidance = guidance_scale > 1.0
        if do_classifier_free_guidance:
            uncond_tokens = [""]
            max_length = self.tokenizer.model_max_length
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
            embeddings = torch.cat([uncond_embeddings, cond_embeddings])
        else:
            embeddings = cond_embeddings

        # Initialize latents
        latents_shape = (1, self.unet.in_channels, height // 8, width // 8)
        latents = torch.randn(latents_shape, dtype=embeddings.dtype).to(self.device)
        torch.manual_seed(seed)

        self.scheduler.set_timesteps(num_inference_steps)
        timesteps_tensor = self.scheduler.timesteps.to(self.device)

        for i, t in tqdm(enumerate(timesteps_tensor)):
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=embeddings)['sample']

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']

        # Decode latents to image space
        latents = 1 / 0.18215 * latents
        images = self.vae.decode(latents).sample
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.detach().cpu().permute(0, 2, 3, 1).float().numpy()

        if output_type == "pil":
            images = (images * 255).round().astype('uint8')
            images = [Image.fromarray(image) for image in images]

        return images

    @torch.no_grad()
    def diffusionloss_IM_text(self,
        text_embeddings: torch.Tensor,
        input_image,
        seed: int = 0,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        resolution: Optional[int] = 512,
        center_crop: bool = False,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: float = 7.5
    ):
        """
        Compute diffusion loss for text embedding alignment for the task of image identification.

        Args:
            text_embeddings: Text embeddings to guide generation.
            input_image: Input image to align with text embeddings.
            Other parameters are similar to `generateImage`.

        Returns:
            Average diffusion loss.
        """
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` must be divisible by 8, but received {height} and {width}.")

        image_transforms = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        init_image = image_transforms(input_image)[None].to(self.device, dtype=torch.float32)
        init_latents = self.vae.encode(init_image).latent_dist.sample()
        init_latents = 0.18215 * init_latents

        self.scheduler.set_timesteps(num_inference_steps)
        timesteps_tensor = self.scheduler.timesteps.to(self.device)
        torch.manual_seed(seed)
        noise = torch.randn_like(init_latents)

        loss_avg = AverageMeter()

        for i, t in tqdm(enumerate(timesteps_tensor)):
            noisy_latents = self.scheduler.add_noise(init_latents, noise, t)
            latent_input = torch.cat([noisy_latents] * 2)
            noise_pred = self.unet(latent_input, t, text_embeddings)['sample']
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            loss_avg.update(loss.detach(), init_latents.size(0))

        return loss_avg

    @torch.no_grad()
    def diffusionloss_IM_IM(self,
        image_ID_embeddings: torch.Tensor,
        image,
        seed: int = 0,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        resolution: Optional[int] = 512,
        center_crop: bool = False,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: float = 7.5
    ):
        """
        Compute diffusion loss for Image_text embedding alignment for the task of image identification.

        Args:
            image_ID_embeddings: Image embeddings to guide generation.
            image: Input image to align with image embeddings.
            Other parameters are similar to `generateImage`.

        Returns:
            Average diffusion loss.
        """
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` must be divisible by 8, but received {height} and {width}.")

        image_transforms = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        init_image = image_transforms(image)[None].to(self.device, dtype=torch.float32)
        init_latents = self.vae.encode(init_image).latent_dist.sample()
        init_latents = 0.18215 * init_latents

        self.scheduler.set_timesteps(num_inference_steps)
        timesteps_tensor = self.scheduler.timesteps.to(self.device)
        torch.manual_seed(seed)
        noise = torch.randn_like(init_latents)

        loss_avg = AverageMeter()

        for i, t in tqdm(enumerate(timesteps_tensor)):
            noisy_latents = self.scheduler.add_noise(init_latents, noise, t)
            noise_pred = self.unet(noisy_latents, t, image_ID_embeddings)['sample']
            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            loss_avg.update(loss.detach(), init_latents.size(0))

        return loss_avg
