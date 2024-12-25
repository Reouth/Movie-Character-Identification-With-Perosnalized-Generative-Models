import torch
from typing import Optional
import os
import gc
from src import DataUpload
from src import HelperFunctions


def multi_image_identifier(imagic_pretrained_path, csv_folder, sd_model_name, clip_model_name, device, image_list,
                                      category_class=False, imagic_pipe=False, alpha=0, seed: int = 0,
                                      height: Optional[int] = 512, width: Optional[int] = 512, resolution: Optional[int] = 512,
                                      num_inference_steps: Optional[int] = 50, guidance_scale: float = 7.5):

    if category_class and not imagic_pipe:
        cat_files = DataUpload.get_category_params(imagic_pipe, imagic_pretrained_path, clip_model_name, sd_model_name, device)
        all_files = list(cat_files.keys())

    else:
        cat_files = {}
        all_files = set(os.listdir(imagic_pretrained_path))

    for file in all_files:
        embeds_files,pipe_name = DataUpload.get_embeds(category_class,imagic_pipe,file,cat_files,alpha,imagic_pretrained_path,sd_model_name,clip_model_name,device)
        csv_dir = os.path.join(csv_folder, pipe_name)
        os.makedirs(csv_dir, exist_ok=True)


        for image_name, image, _ in image_list:
            cls = image_name.rsplit("_", 1)[0]
            df, csv_file_path = HelperFunctions.get_current_csv(str(csv_dir), cls)
            image_flag = HelperFunctions.row_exist(df,cls,file)
            if not image_flag:
                loss = conditioned_diffusion_identifier(
                    embeds_files, image, seed, height, width, resolution, num_inference_steps, guidance_scale
                )
                HelperFunctions.save_to_csv(loss, df, image_name, csv_file_path)

def conditioned_diffusion_identifier(parameters, test_image, seed: int = 0, height: Optional[int] = 512, width: Optional[int] = 512,
                            resolution: Optional[int] = 512, num_inference_steps: Optional[int] = 50,
                            guidance_scale: float = 7.5):


    gc.collect()  # Collect garbage to free memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    loss = {}

    for embeds_name,params in parameters.items():
        pipeline, embeddings = params
        with torch.autocast("cuda"), torch.inference_mode():
            loss_avg = pipeline.loss_image_image(
                cond_embeddings=embeddings,
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



