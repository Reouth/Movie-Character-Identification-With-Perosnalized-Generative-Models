# **SD_model.py**

"""
This script provides functions for diffusion-based generation and identification models.
"""

import torch
from typing import Optional
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler
from diffusers.utils import logging
from transformers import CLIPTextModel, CLIPTokenizer

import os
import gc
from src import data_upload
from src import helper_functions
from src import SD_pipeline

def SD_pretrained_load(SD_MODEL_NAME, CLIP_MODEL_NAME, device, imagic_trained=False):
    """
    Load pretrained components for the Stable Diffusion model.

    Args:
        SD_MODEL_NAME (str): Path to the Stable Diffusion model.
        CLIP_MODEL_NAME (str): Path to the CLIP model.
        device (str): Device to load the models on.
        imagic_trained (bool): Whether to load Imagic-specific paths.

    Returns:
        tuple: VAE, text encoder, tokenizer, UNet, and scheduler components.
    """
    if imagic_trained:
        vae_path = os.path.join(SD_MODEL_NAME, 'vae')
        tokenizer_path = os.path.join(SD_MODEL_NAME, 'tokenizer')
        text_encoder_path = os.path.join(SD_MODEL_NAME, 'text_encoder')
        unet_path = os.path.join(SD_MODEL_NAME, 'unet')
    else:
        vae_path = unet_path = SD_MODEL_NAME
        tokenizer_path = text_encoder_path = CLIP_MODEL_NAME

    vae = AutoencoderKL.from_pretrained(vae_path, subfolder='vae', token=True).to(device)
    tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)
    text_encoder = CLIPTextModel.from_pretrained(text_encoder_path).to(device)
    unet = UNet2DConditionModel.from_pretrained(unet_path, subfolder='unet', token=True).to(device)

    scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False
    )

    logger = logging.get_logger(__name__)
    return vae, text_encoder, tokenizer, unet, scheduler

def all_generator(all_files, output_folder, imagic_pretrained_path, Imagic_pipe, SD_model_name, CLIP_model_name, device,
                  seed_range, alpha_range, guidance_scale_range, cat_embeds=None, height: Optional[int] = 512,
                  width: Optional[int] = 512, num_inference_steps: Optional[int] = 50):
    """
    Generate images for all embeddings and parameter combinations.

    Args:
        all_files (list): List of embedding files.
        output_folder (str): Path to save generated images.
        imagic_pretrained_path (str): Path to pretrained Imagic models.
        Imagic_pipe (bool): Whether to use Imagic pipeline.
        SD_model_name (str): Path to Stable Diffusion model.
        CLIP_model_name (str): Path to CLIP model.
        device (str): Device to run the pipeline.
        seed_range (range): Range of seed values.
        alpha_range (range): Range of alpha values.
        guidance_scale_range (range): Range of guidance scale values.
        cat_embeds (optional, dict): Concatenated embeddings for specific categories.
        height (int): Height of generated images.
        width (int): Width of generated images.
        num_inference_steps (int): Number of inference steps.
    """
    for embeds in all_files:
        if cat_embeds is not None:
            cat_embeddings = cat_embeds[embeds][0] + cat_embeds[embeds][1] / cat_embeds[embeds][2]
            Imagic_pipe = False
        else:
            cat_embeddings = None

        if Imagic_pipe:
            pipe_name = 'Imagic_pipeline'
            SD_pretrained_model = None
        else:
            pipe_name = 'SD_pipeline'
            SD_pretrained_model = SD_pretrained_load(SD_model_name, CLIP_model_name, device)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        imagic_parameters = data_upload.upload_single_imagic_params(
            imagic_pretrained_path, embeds, CLIP_model_name, device, Imagic_pipe
        )

        for seed in seed_range:
            output_dir = os.path.join(output_folder, pipe_name, f"seed_{seed}")
            os.makedirs(output_dir, exist_ok=True)
            for alpha in alpha_range:
                for guidance_scale in guidance_scale_range:
                    image_checkpoint, image_path, embeds_file = helper_functions.generated_image_checkpoint(
                        output_dir, embeds, alpha, guidance_scale
                    )
                    if image_checkpoint:
                        continue

                    image_generator(
                        os.path.dirname(image_path),
                        imagic_parameters,
                        os.path.basename(image_path),
                        cat_embeddings,
                        SD_pretrained_model,
                        alpha,
                        seed=seed,
                        height=height,
                        width=width,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale
                    )

def image_generator(output_folder, imagic_parameters, image_name, cat_embeds=None, SD_pretrained_models=None, alpha=0,
                     seed: int = 0, height: Optional[int] = 512, width: Optional[int] = 512,
                     num_inference_steps: Optional[int] = 50, guidance_scale: float = 7.5):
    """
    Generate a single image using the specified parameters.

    Args:
        output_folder (str): Directory to save the image.
        imagic_parameters (tuple): Parameters from Imagic pipeline.
        image_name (str): Name of the output image.
        cat_embeds (optional, torch.Tensor): Concatenated embeddings for the image.
        SD_pretrained_models (optional, tuple): Pretrained Stable Diffusion components.
        alpha (float): Alpha blending parameter for embeddings.
        seed (int): Random seed for reproducibility.
        height (int): Height of the generated image.
        width (int): Width of the generated image.
        num_inference_steps (int): Number of diffusion steps.
        guidance_scale (float): Guidance scale for classifier-free guidance.
    """
    if SD_pretrained_models is not None:
        pipeline = SD_pipeline.StableDiffusionPipeline(*SD_pretrained_models)
        _, target_embeddings, optimized_embeddings = imagic_parameters
    else:
        pipeline, target_embeddings, optimized_embeddings = imagic_parameters

    if cat_embeds is not None:
        embeddings = cat_embeds
    else:
        embeddings = alpha * target_embeddings + (1 - alpha) * optimized_embeddings

    with torch.autocast("cuda"), torch.inference_mode():
        images = pipeline.generateImage(
            cond_embeddings=embeddings,
            seed=seed,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )

    images[0].save(os.path.join(output_folder, image_name))

def conditioned_classifier(parameters, test_image, seed: int = 0, height: Optional[int] = 512, width: Optional[int] = 512,
                            resolution: Optional[int] = 512, num_inference_steps: Optional[int] = 50,
                            guidance_scale: float = 7.5):
    """
    Compute the diffusion loss for a conditioned classifier.

    Args:
        parameters (dict): Parameters containing the pipeline and embeddings.
        test_image (PIL.Image): Image to compute the loss for.
        Other arguments specify generation and guidance parameters.

    Returns:
        dict: Loss values for each embedding.
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

def all_embeds_conditioned_classifier(imagic_pretrained_path, csv_folder, SD_model_name, CLIP_model_name, device, image_list,
                                      category_class=False, Imagic_pipe=False, alpha=0, seed: int = 0,
                                      height: Optional[int] = 512, width: Optional[int] = 512, resolution: Optional[int] = 512,
                                      num_inference_steps: Optional[int] = 50, guidance_scale: float = 7.5):
    """
    Apply a conditioned classifier to all embeddings and save the results.

    Args:
        imagic_pretrained_path (str): Path to Imagic embeddings.
        csv_folder (str): Directory to save CSV results.
        Other arguments specify generation and guidance parameters.
    """
    if Imagic_pipe:
        pipe_name = 'Imagic_pipeline'
        SD_pretrained_model = None
        all_files = set(os.listdir(imagic_pretrained_path))
    else:
        pipe_name = 'SD_pipeline'
        SD_pretrained_model = SD_pretrained_load(SD_model_name, CLIP_model_name, device)
        if category_class:
            pipe_name = 'SD_embeds_cat_avg'
            cat_files = data_upload.upload_cat_embeds(imagic_pretrained_path, CLIP_model_name, device, Imagic_pipe, SD_pretrained_model)
            all_files = list(cat_files.keys())
        else:
            all_files = set(os.listdir(imagic_pretrained_path))

    for file in all_files:
        if category_class:
            embeds_files = {file: cat_files[file]}
        else:
            embeds_files = data_upload.upload_embeds(
                imagic_pretrained_path, file, CLIP_model_name, alpha, device, Imagic_pipe, SD_pretrained_model
            )

        csv_dir = os.path.join(csv_folder, pipe_name)
        os.makedirs(csv_dir, exist_ok=True)

        for image_name, image, _ in image_list:
            cls = image_name.rsplit("_", 1)[0]
            image_flag, df_sd, csv_file_path = helper_functions.csv_checkpoint(
                csv_dir, cls, image_name, file
            )
            if not image_flag:
                SD_loss = conditioned_classifier(
                    embeds_files, image, seed, height, width, resolution, num_inference_steps, guidance_scale
                )
                helper_functions.save_to_csv(SD_loss, df_sd, image_name, csv_file_path)

if __name__ == "__main__":
    print("This module contains Stable Diffusion and Imagic utilities for image generation and classification.")
