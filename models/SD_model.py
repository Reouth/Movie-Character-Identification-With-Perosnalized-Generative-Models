# **SD_model.py**

"""
This script provides functions for loading and managing models, generating images, and performing image-conditioned classification tasks
using Stable Diffusion and Imagic frameworks.
"""

import os
import gc
import torch
from typing import List, Optional, Union
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import DDIMScheduler
from diffusers.utils import logging
from transformers import CLIPTextModel, CLIPTokenizer
import data_upload
import helper_functions
import new_SD_pipeline

def SD_pretrained_load(SD_MODEL_NAME: str, CLIP_MODEL_NAME: str, device: str, imagic_trained: bool = False):
    """
    Loads the necessary pretrained components for Stable Diffusion.

    Args:
        SD_MODEL_NAME (str): Path to the Stable Diffusion model.
        CLIP_MODEL_NAME (str): Path to the CLIP model.
        device (str): Device to load the models onto.
        imagic_trained (bool): Whether to use Imagic-trained models.

    Returns:
        Tuple: VAE, text encoder, tokenizer, UNet, and scheduler components.
    """
    vae_path = os.path.join(SD_MODEL_NAME, 'vae') if imagic_trained else SD_MODEL_NAME
    unet_path = os.path.join(SD_MODEL_NAME, 'unet') if imagic_trained else SD_MODEL_NAME
    tokenizer_path = os.path.join(SD_MODEL_NAME, 'tokenizer') if imagic_trained else CLIP_MODEL_NAME
    text_encoder_path = os.path.join(SD_MODEL_NAME, 'text_encoder') if imagic_trained else CLIP_MODEL_NAME

    vae = AutoencoderKL.from_pretrained(vae_path, subfolder='vae', token=True).to(device)
    tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)
    text_encoder = CLIPTextModel.from_pretrained(text_encoder_path).to(device)
    unet = UNet2DConditionModel.from_pretrained(unet_path, subfolder='unet', token=True).to(device)
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False)

    logger = logging.get_logger(__name__)
    return vae, text_encoder, tokenizer, unet, scheduler

def generate_images(
    output_folder: str, imagic_parameters: tuple, image_name: str, cat_embeds=None, SD_pretrained_models=None,
    alpha: float = 0, seed: int = 0, height: int = 512, width: int = 512, num_inference_steps: int = 50,
    guidance_scale: float = 7.5
):
    """
    Generates images using Imagic or Stable Diffusion pipelines.

    Args:
        output_folder (str): Directory to save the generated images.
        imagic_parameters (tuple): Embedding parameters from Imagic.
        image_name (str): Name of the output image.
        cat_embeds (torch.Tensor, optional): Category embeddings.
        SD_pretrained_models (tuple, optional): Pretrained models for Stable Diffusion.
        alpha (float): Alpha blending parameter for embeddings.
        seed (int): Random seed for generation.
        height (int): Image height.
        width (int): Image width.
        num_inference_steps (int): Number of inference steps.
        guidance_scale (float): Scale for classifier-free guidance.
    """
    pipeline = (
        new_SD_pipeline.StableDiffusionPipeline(*SD_pretrained_models)
        if SD_pretrained_models else imagic_parameters[0]
    )
    target_embeddings, optimized_embeddings = imagic_parameters[1], imagic_parameters[2]
    embeddings = cat_embeds if cat_embeds else alpha * target_embeddings + (1 - alpha) * optimized_embeddings

    print(f"Generating image {image_name}")
    with torch.autocast("cuda"), torch.inference_mode():
        images = pipeline.generateImage(
            cond_embeddings=embeddings,
            seed=seed,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
    image = images[0]
    image.save(os.path.join(output_folder, image_name))

def conditioned_classifier(
    parameters: dict, test_image, seed: int = 0, height: int = 512, width: int = 512, resolution: int = 512,
    num_inference_steps: int = 50, guidance_scale: float = 7.5
):
    """
    Performs classification on test images conditioned on embeddings.

    Args:
        parameters (dict): Embedding parameters.
        test_image (PIL.Image): Input test image.
        seed (int): Random seed for generation.
        height (int): Image height.
        width (int): Image width.
        resolution (int): Image resolution.
        num_inference_steps (int): Number of inference steps.
        guidance_scale (float): Scale for classifier-free guidance.

    Returns:
        dict: Classification loss for each embedding.
    """
    SD_loss = {}
    for embeds_name, params in parameters.items():
        pipeline, embeddings = params
        with torch.autocast("cuda"), torch.inference_mode():
            loss_avg = pipeline.diffusionloss_IM_IM(
                image_ID_embeddings=embeddings,
                image=test_image.convert('RGB'),
                seed=seed,
                height=height,
                width=width,
                resolution=resolution,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            )
        SD_loss[embeds_name] = loss_avg.avg.item()
    return SD_loss

if __name__ == "__main__":
    print("This module contains Stable Diffusion and Imagic utilities for image generation and classification.")
