import os
from PIL import Image
from src import SDModel
import torch
import gc
import pathlib
import numpy as np


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

def upload_csvs(csv_dir_path):

    csv_dir = pathlib.Path(csv_dir_path)
    csv_paths = list(csv_dir.glob("*.csv"))  # Collect all CSV files with .csv extension
    return csv_paths

def is_image(file_path):

    try:
        with Image.open(file_path) as img:
            img.verify()  # Verify the image integrity
            return True
    except (IOError, SyntaxError) as e:
        print(f"File {file_path} is not an image or cannot be opened. Error: {e}")
        return False

def upload_images(base_path, class_batch=float('inf'), max_frames=float('inf')):
    """
    Recursively load images from a directory and rename them based on their folder.

    Args:
        base_path (str): Path to the base directory containing images.
        class_batch (int): Maximum number of images per class (default is no limit).
        max_frames (int): Maximum total number of images to load (default is no limit).

    Returns:
        list: List of tuples containing image name, image object, and original path.
    """
    image_data = []
    image_counts = {}
    total_frames_count = 0

    sorted_items = sorted(os.listdir(base_path))

    sample_tf = sample_batch_size(class_batch,sorted_items,base_path)

    for i, item in enumerate(sorted_items):
        item_path = os.path.join(base_path, item)

        if os.path.isdir(item_path):
            print(f"Entering directory: {item_path}")
            # Recursively process the directory
            image_data += upload_images(item_path, class_batch, max_frames)
        elif is_image(item_path):
            if i % sample_tf != 0:
                continue

            folder_name = os.path.basename(base_path)  # Get the folder name
            if folder_name not in image_counts:
                image_counts[folder_name] = 0

            if total_frames_count >= max_frames or image_counts[folder_name] >= class_batch:
                print(f"{image_counts[folder_name]} frames in {folder_name} class")
                return image_data

            image_counts[folder_name] += 1
            total_frames_count += 1
            frame_number = image_counts[folder_name]
            new_name = f"{folder_name}_{frame_number:03d}.jpg"

            # Store the image data
            image_data.append((new_name, Image.open(item_path), item_path))
        else:
            print(f"Not an image file: {item_path}")

    return image_data

def sample_batch_size(class_batch,sorted_items,base_path):
    if class_batch != float('inf'):
        sample_tf = np.floor(
            len([item for item in sorted_items if is_image(os.path.join(base_path, item))]) / class_batch)
        sample_tf = int(max(sample_tf, 1))  # Ensure at least one image is sampled
    else:
        sample_tf = 1
    return sample_tf







