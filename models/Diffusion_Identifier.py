import torch
from typing import Optional
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler
from diffusers.utils import logging
from torchvision.models.detection.roi_heads import paste_mask_in_image
from transformers import CLIPTextModel, CLIPTokenizer

import os
import gc
from src import data_upload
from src import helper_functions
from src import SD_pipeline


def multi_image_identifier(imagic_pretrained_path, csv_folder, sd_model_name, clip_model_name, device, image_list,
                                      category_class=False, imagic_pipe=False, alpha=0, seed: int = 0,
                                      height: Optional[int] = 512, width: Optional[int] = 512, resolution: Optional[int] = 512,
                                      num_inference_steps: Optional[int] = 50, guidance_scale: float = 7.5):

    if category_class and not imagic_pipe:
        cat_files = get_category_params(imagic_pipe, imagic_pretrained_path, clip_model_name, sd_model_name, device)
        all_files = list(cat_files.keys())

    else:
        cat_files = {}
        all_files = set(os.listdir(imagic_pretrained_path))

    for file in all_files:
        embeds_files,pipe_name = get_embeds(category_class,imagic_pipe,file,cat_files,alpha,imagic_pretrained_path,sd_model_name,clip_model_name)
        csv_dir = os.path.join(csv_folder, pipe_name)
        os.makedirs(csv_dir, exist_ok=True)
        df,csv_file_path = helper_functions.get_current_csv(csv_dir,file)

        for image_name, image, _ in image_list:
            cls = image_name.rsplit("_", 1)[0]
            image_flag = helper_functions.row_exist(df,cls,file)

            if not image_flag:
                loss = conditioned_diffusion_identifier(
                    embeds_files, image, seed, height, width, resolution, num_inference_steps, guidance_scale
                )
                helper_functions.save_to_csv(loss, df, image_name, csv_file_path)

def conditioned_diffusion_identifier(parameters, test_image, seed: int = 0, height: Optional[int] = 512, width: Optional[int] = 512,
                            resolution: Optional[int] = 512, num_inference_steps: Optional[int] = 50,
                            guidance_scale: float = 7.5):


    gc.collect()  # Collect garbage to free memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    loss = {}

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
        loss[embeds_name] = loss_avg.avg.item()

    return loss


def get_category_params(imagic_pipe,imagic_pretrained_path,clip_model_name,sd_model_name,device):
    sd_pretrained_model = SD_pipeline.SD_pretrained_load(sd_model_name, clip_model_name, device)
    cat_files = data_upload.upload_cat_embeds(imagic_pretrained_path, clip_model_name, device, imagic_pipe, sd_pretrained_model)
    return cat_files


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

def get_embeds(category_class,imagic_pipe,file,cat_files,alpha,imagic_pretrained_path,sd_model_name,clip_model_name):
    embeds_files = {}
    if category_class and not imagic_pipe:
         embeds_files = {file:cat_files[file]}
         pipe_name = 'SD_embeds_cat_avg'
    else:
        parameters, pipe_name = get_pretrained_params(imagic_pipe, imagic_pretrained_path,file, sd_model_name,
                                                      clip_model_name)
        pipeline,target_embeddings, optimized_embeddings = parameters
        embeddings = alpha * target_embeddings + (1 - alpha) * optimized_embeddings
        embeds_files[file] = (pipeline, embeddings)
    return embeds_files, pipe_name

