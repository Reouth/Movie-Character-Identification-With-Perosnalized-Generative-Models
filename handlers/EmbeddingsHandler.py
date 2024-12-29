import os
from handlers import SDModel
import torch
import gc


def get_pretrained_params(imagic_pipe,imagic_pretrained_path,sd_model_name,clip_model_name,embeds,device):
    imagic_parameters = upload_single_imagic_params(
        imagic_pretrained_path, embeds, clip_model_name, device, imagic_pipe)
    if not imagic_pipe:
        sd_pretrained_model = SDModel.sd_pretrained_load(sd_model_name, clip_model_name, device)
        pipeline = SDModel.StableDiffusionPipeline(*sd_pretrained_model)
        parameters = (pipeline, imagic_parameters[1],imagic_parameters[2])
        pipe_name = 'SD_pipeline'
    else:
        parameters = imagic_parameters
        pipe_name = 'Imagic_pipeline'
    return parameters, pipe_name

def get_category_params(imagic_pipe,imagic_pretrained_path,clip_model_name,sd_model_name,device):
    sd_pretrained_model = SDModel.sd_pretrained_load(sd_model_name, clip_model_name, device)
    cat_files = upload_cat_embeds(imagic_pretrained_path, clip_model_name, device, imagic_pipe, sd_pretrained_model)
    return cat_files


def get_embeds(category_class,imagic_pipe,file,cat_files,alpha,imagic_pretrained_path,sd_model_name,clip_model_name,device):
    embeds_files = {}
    if category_class and not imagic_pipe:
         embeds_files = {file:cat_files[file]}
         pipe_name = 'SD_embeds_cat_avg'
    else:
        parameters, pipe_name = get_pretrained_params(imagic_pipe, imagic_pretrained_path, sd_model_name,
                                                      clip_model_name,file,device)
        pipeline,target_embeddings, optimized_embeddings = parameters
        embeddings = alpha * target_embeddings + (1 - alpha) * optimized_embeddings
        embeds_files[file] = (pipeline, embeddings)
    return embeds_files, pipe_name

def upload_single_imagic_params(path, embeds_file, clip_model_name, device, imagic_pipe):
    """
    Load Imagic model parameters and embeddings from a specified directory.

    Args:
        path (str): Base path to the embeddings directory.
        embeds_file (str): Specific embeddings file to load.
        clip_model_name (str): Name of the CLIP model.
        device (torch.device): Device to use for loading embeddings.
        imagic_pipe (bool): Whether to initialize an Imagic pipeline.

    Returns:
        tuple: (Pipeline object, target embeddings tensor, optimized embeddings tensor)
    """
    imagic_pretrained_path = os.path.join(path, embeds_file)
    if os.path.isdir(imagic_pretrained_path):
        print(f"Uploading embeddings for directory: {imagic_pretrained_path}")
        if imagic_pipe:
            # Load pretrained models
            pretrained_models = SDModel.sd_pretrained_load(imagic_pretrained_path, clip_model_name, device, True)
            pipeline = SDModel.StableDiffusionPipeline(*pretrained_models)
        else:
            pipeline = None

        # Load target and optimized embeddings
        target_embeddings = torch.load(os.path.join(imagic_pretrained_path, "target_embeddings.pt")).to(device)
        optimized_embeddings = torch.load(os.path.join(imagic_pretrained_path, "optimized_embeddings.pt")).to(device)

        return pipeline, target_embeddings, optimized_embeddings
    else:
        print(f"No embedding directory called {imagic_pretrained_path}")

def upload_cat_embeds(path, clip_model_name, device, imagic_pipe, sd_pipe):
    """
    Aggregate embeddings for categories by averaging across multiple embeddings.

    Args:
        path (str): Path to directory containing embeddings.
        clip_model_name (str): Name of the CLIP model.
        device (torch.device): Device for embedding tensors.
        imagic_pipe (bool): Whether to use Imagic pipeline.
        sd_pipe (list): Pretrained SD pipeline components.

    Returns:
        dict: Dictionary with categories as keys and (Pipeline, Average Embedding) as values.
    """
    final_embeds = {}
    embeddings = {}
    pipeline = SDModel.StableDiffusionPipeline(*sd_pipe)

    for embeds in os.listdir(path):
        embeds_category = embeds.rsplit("_", 1)[0]  # Extract category from file name

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        imagic_params = upload_single_imagic_params(path, embeds, clip_model_name, device, imagic_pipe)
        _, target_embeddings, optimized_embeddings = imagic_params

        if embeds_category in embeddings:
            existing_target_embeds, existing_optimized_embeds, count = embeddings[embeds_category]
            embeddings[embeds_category] = (
                existing_target_embeds + target_embeddings,
                existing_optimized_embeds + optimized_embeddings,
                count + 1
            )
        else:
            embeddings[embeds_category] = (target_embeddings, optimized_embeddings, 1)

    for cat, params in embeddings.items():
        total_target_embeds, total_optimized_embeds, count = params
        avg_embedding = (total_target_embeds + total_optimized_embeds) / count
        final_embeds[cat] = (pipeline, avg_embedding)

    return final_embeds
