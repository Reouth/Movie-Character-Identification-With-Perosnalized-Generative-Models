import os
from PIL import Image
from models import Diffusion_generate
from src import SD_pipeline
import torch
import gc
import pathlib
import numpy as np

def upload_csvs(csv_dir_path):
    """
    Load all CSV files from a specified directory.

    Args:
        csv_dir_path (str): Path to the directory containing CSV files.

    Returns:
        list: List of paths to the CSV files found in the directory.
    """
    csv_dir = pathlib.Path(csv_dir_path)
    csv_paths = list(csv_dir.glob("*.csv"))  # Collect all CSV files with .csv extension
    return csv_paths

def is_image(file_path):
    """
    Check if a file is a valid image.

    Args:
        file_path (str): Path to the file.

    Returns:
        bool: True if the file is an image, False otherwise.
    """
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
    image_data = []  # Store image data and metadata
    image_counts = {}  # Track counts of images per folder
    total_frames_count = 0  # Total count of loaded images

    # Ensure a deterministic order by sorting directory contents
    sorted_items = sorted(os.listdir(base_path))

    # Determine sampling frequency if a class_batch limit is set
    if class_batch != float('inf'):
        sample_tf = np.floor(
            len([item for item in sorted_items if is_image(os.path.join(base_path, item))]) / class_batch)
        sample_tf = int(max(sample_tf, 1))  # Ensure at least one image is sampled
    else:
        sample_tf = 1

    for i, item in enumerate(sorted_items):
        item_path = os.path.join(base_path, item)

        if os.path.isdir(item_path):  # If the item is a directory
            print(f"Entering directory: {item_path}")
            # Recursively process the directory
            image_data += upload_images(item_path, class_batch, max_frames)
        elif is_image(item_path):  # If the item is a valid image
            if i % sample_tf != 0:  # Skip images based on the sampling frequency
                continue

            folder_name = os.path.basename(base_path)  # Get the folder name

            if folder_name not in image_counts:
                image_counts[folder_name] = 0

            # Stop if limits for total frames or class images are reached
            if total_frames_count >= max_frames or image_counts[folder_name] >= class_batch:
                print(f"{image_counts[folder_name]} frames in {folder_name} class")
                return image_data

            image_counts[folder_name] += 1
            total_frames_count += 1

            # Create a new name for the image using zero-padded numbering
            frame_number = image_counts[folder_name]
            new_name = f"{folder_name}_{frame_number:03d}.jpg"

            # Store the image data
            image_data.append((new_name, Image.open(item_path), item_path))
        else:
            print(f"Not an image file: {item_path}")

    return image_data

def upload_single_imagic_params(path, embeds_file, CLIP_model_name, device, Imagic_pipe):
    """
    Load Imagic model parameters and embeddings from a specified directory.

    Args:
        path (str): Base path to the embeddings directory.
        embeds_file (str): Specific embeddings file to load.
        CLIP_model_name (str): Name of the CLIP model.
        device (torch.device): Device to use for loading embeddings.
        Imagic_pipe (bool): Whether to initialize an Imagic pipeline.

    Returns:
        tuple: (Pipeline object, target embeddings tensor, optimized embeddings tensor)
    """
    imagic_pretrained_path = os.path.join(path, embeds_file)
    if os.path.isdir(imagic_pretrained_path):
        print(f"Uploading embeddings for directory: {imagic_pretrained_path}")
        if Imagic_pipe:
            # Load pretrained models
            pretrained_models = SD_pipeline.SD_pretrained_load(imagic_pretrained_path, CLIP_model_name, device, True)
            pipeline = SD_pipeline.StableDiffusionPipeline(*pretrained_models)
        else:
            pipeline = None

        # Load target and optimized embeddings
        target_embeddings = torch.load(os.path.join(imagic_pretrained_path, "target_embeddings.pt")).to(device)
        optimized_embeddings = torch.load(os.path.join(imagic_pretrained_path, "optimized_embeddings.pt")).to(device)

        return pipeline, target_embeddings, optimized_embeddings
    else:
        print(f"No embedding directory called {imagic_pretrained_path}")

def upload_embeds(path, file, CLIP_model_name, alpha, device, imagic_pipe, SD_pretrained=None):
    """
    Upload and blend embeddings using a weighted combination of target and optimized embeddings.

    Args:
        path (str): Base directory for embeddings.
        file (str): Specific file to process.
        CLIP_model_name (str): Name of the CLIP model.
        alpha (float): Weight for blending embeddings.
        device (torch.device): Device for embedding tensors.
        imagic_pipe (bool): Whether to use the Imagic pipeline.
        SD_pretrained (list, optional): Pretrained SD pipeline components.

    Returns:
        dict: Dictionary with file names as keys and (Pipeline, Embeddings) as values.
    """
    all_embeds = {}
    gc.collect()  # Collect garbage to free memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Clear GPU memory

    imagic_parameters = upload_single_imagic_params(path, file, CLIP_model_name, device, imagic_pipe)

    if SD_pretrained:
        pipeline = SD_pipeline.StableDiffusionPipeline(*SD_pretrained)
        _, target_embeddings, optimized_embeddings = imagic_parameters
    else:
        pipeline, target_embeddings, optimized_embeddings = imagic_parameters

    # Blend embeddings using the alpha factor
    embeddings = alpha * target_embeddings + (1 - alpha) * optimized_embeddings
    all_embeds[file] = (pipeline, embeddings)

    return all_embeds

def upload_cat_embeds(path, CLIP_model_name, device, imagic_pipe, SD_pipe):
    """
    Aggregate embeddings for categories by averaging across multiple embeddings.

    Args:
        path (str): Path to directory containing embeddings.
        CLIP_model_name (str): Name of the CLIP model.
        device (torch.device): Device for embedding tensors.
        imagic_pipe (bool): Whether to use Imagic pipeline.
        SD_pipe (list): Pretrained SD pipeline components.

    Returns:
        dict: Dictionary with categories as keys and (Pipeline, Average Embedding) as values.
    """
    final_embeds = {}
    embeddings = {}
    pipeline = SD_pipeline.StableDiffusionPipeline(*SD_pipe)

    for embeds in os.listdir(path):
        embeds_category = embeds.rsplit("_", 1)[0]  # Extract category from file name

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        imagic_params = upload_single_imagic_params(path, embeds, CLIP_model_name, device, imagic_pipe)
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
