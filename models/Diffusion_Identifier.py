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


def multi_image_identifier(imagic_pretrained_path, csv_folder, sd_model_name, clip_model_name, device, image_list,
                                      category_class=False, Imagic_pipe=False, alpha=0, seed: int = 0,
                                      height: Optional[int] = 512, width: Optional[int] = 512, resolution: Optional[int] = 512,
                                      num_inference_steps: Optional[int] = 50, guidance_scale: float = 7.5):
    parameters, pipe_name = get_pretrained_params(imagic_pipe, imagic_pretrained_path, sd_model_name, clip_model_name,
                                                  embeds, device)
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


def get_pretrained_params(imagic_pipe,imagic_pretrained_path,sd_model_name,clip_model_name,embeds,device):
    imagic_parameters = data_upload.upload_single_imagic_params(
        imagic_pretrained_path, embeds, clip_model_name, device, imagic_pipe)
    if not imagic_pipe:
        sd_pretrained_model = SD_pipeline.SD_pretrained_load(sd_model_name, clip_model_name, device)
        pipeline = SD_pipeline.StableDiffusionPipeline(*sd_pretrained_model)
        parameters = (pipeline, imagic_parameters[1],imagic_parameters[2])
        pipe_name = 'SD_pipeline'
    else:
        parameters = imagic_parameters
        pipe_name = 'Imagic_pipeline'
    return parameters, pipe_name