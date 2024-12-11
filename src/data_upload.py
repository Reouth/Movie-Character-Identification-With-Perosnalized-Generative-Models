# **data_upload.py**

"""
This script handles the uploading and organization of image datasets, CSV files, and embeddings for the Imagic framework.
All functionalities from the original script are preserved and optimized for readability and maintainability.
"""

import os
import glob
from typing import List, Tuple, Dict
import csv
import torch
from PIL import Image


def is_image(file_path: str) -> bool:
    """
    Checks if a file is a valid image.

    Args:
        file_path (str): Path to the file.

    Returns:
        bool: True if the file is an image, False otherwise.
    """
    supported_formats = {".jpg", ".jpeg", ".png", ".bmp"}
    return any(file_path.lower().endswith(ext) for ext in supported_formats)


def upload_images(folder_path: str) -> List[Tuple[str, Image.Image, str]]:
    """
    Loads images recursively from a directory and returns their metadata.

    Args:
        folder_path (str): Path to the directory containing images.

    Returns:
        List[Tuple[str, Image.Image, str]]: A list of tuples containing the image name, image object, and full path.
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The specified folder does not exist: {folder_path}")

    images = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if is_image(file):
                full_path = os.path.join(root, file)
                try:
                    image = Image.open(full_path).convert("RGB")
                    images.append((file, image, full_path))
                except Exception as e:
                    print(f"Error loading image {file}: {e}")

    if not images:
        raise ValueError(f"No valid images found in folder: {folder_path}")

    print(f"Loaded {len(images)} images from {folder_path}")
    return images


def upload_csvs(folder_path: str) -> List[str]:
    """
    Loads CSV file paths from a directory.

    Args:
        folder_path (str): Path to the directory containing CSV files.

    Returns:
        List[str]: List of paths to CSV files.
    """
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

    if not csv_files:
        raise ValueError(f"No CSV files found in folder: {folder_path}")

    print(f"Loaded {len(csv_files)} CSV files from {folder_path}")
    return csv_files


def load_csv(file_path: str) -> List[Dict[str, str]]:
    """
    Reads a CSV file and returns its content as a list of dictionaries.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        List[Dict[str, str]]: List of rows as dictionaries.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The specified CSV file does not exist: {file_path}")

    with open(file_path, mode="r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        data = [row for row in reader]

    print(f"Loaded {len(data)} rows from {file_path}")
    return data


def upload_single_imagic_params(folder_path: str) -> Dict[str, torch.Tensor]:
    """
    Loads Imagic embedding parameters from a folder.

    Args:
        folder_path (str): Path to the folder containing Imagic embedding files.

    Returns:
        Dict[str, torch.Tensor]: Dictionary containing pipeline and embedding tensors.
    """
    required_files = ["optimized.pt", "target.pt", "vae"]
    if not all(os.path.exists(os.path.join(folder_path, file)) for file in required_files):
        raise FileNotFoundError(f"Missing required Imagic files in folder: {folder_path}")

    params = {
        "optimized": torch.load(os.path.join(folder_path, "optimized.pt"), map_location=torch.device("cpu")),
        "target": torch.load(os.path.join(folder_path, "target.pt"), map_location=torch.device("cpu"))
    }
    print(f"Loaded Imagic parameters from {folder_path}")
    return params


def upload_embeds(folder_path: str) -> Dict[str, torch.Tensor]:
    """
    Combines and processes embeddings for a given folder.

    Args:
        folder_path (str): Path to the folder containing embeddings.

    Returns:
        Dict[str, torch.Tensor]: Dictionary with alpha-blended embeddings.
    """
    params = upload_single_imagic_params(folder_path)
    alpha = 0.5
    combined_embeds = (alpha * params["target"]) + ((1 - alpha) * params["optimized"])
    params["combined"] = combined_embeds
    print(f"Processed embeddings for folder: {folder_path}")
    return params


def upload_cat_embeds(folder_path: str) -> Dict[str, torch.Tensor]:
    """
    Aggregates embeddings by category, averaging embeddings within the same category.

    Args:
        folder_path (str): Path to the folder containing categorized embeddings.

    Returns:
        Dict[str, torch.Tensor]: Dictionary of averaged embeddings by category.
    """
    category_embeds = {}
    for category in os.listdir(folder_path):
        category_path = os.path.join(folder_path, category)
        if os.path.isdir(category_path):
            embeds = [upload_embeds(os.path.join(category_path, embed)) for embed in os.listdir(category_path)]
            category_embeds[category] = torch.mean(torch.stack([embed["combined"] for embed in embeds]), dim=0)

    print(f"Processed embeddings for {len(category_embeds)} categories.")
    return category_embeds


if __name__ == "__main__":
    # Example usage
    images_folder = "path_to_images"
    csv_folder = "path_to_csvs"
    embeds_folder = "path_to_embeds"

    try:
        images = upload_images(images_folder)
        print(f"Uploaded {len(images)} images.")

        csvs = upload_csvs(csv_folder)
        print(f"Uploaded {len(csvs)} CSV files.")

        embeds = upload_cat_embeds(embeds_folder)
        print(f"Processed {len(embeds)} categorized embeddings.")

    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
