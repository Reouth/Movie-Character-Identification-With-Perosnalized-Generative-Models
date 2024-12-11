# **data_upload.py**

"""
This script manages the uploading and organization of image datasets for the Imagic framework.
It provides utilities to load images from a directory, validate inputs, and prepare them for embedding creation.
"""

import os
from typing import List, Tuple


def upload_images(folder_path: str) -> List[Tuple[str, str, str]]:
    """
    Reads and returns the paths of all images in the specified folder.

    Args:
        folder_path (str): Path to the folder containing image files.

    Returns:
        List[Tuple[str, str, str]]: A list of tuples, each containing the file name, folder path, and full path of an image.

    Raises:
        FileNotFoundError: If the folder_path does not exist.
        ValueError: If the folder_path contains no image files.
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The specified folder does not exist: {folder_path}")

    images = []
    supported_formats = {".jpg", ".jpeg", ".png", ".bmp"}
    
    for root, _, files in os.walk(folder_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in supported_formats):
                full_path = os.path.join(root, file)
                images.append((file, root, full_path))

    if not images:
        raise ValueError(f"No supported image files found in the folder: {folder_path}")

    print(f"Loaded {len(images)} images from {folder_path}")
    return images


def validate_image_paths(image_paths: List[str]) -> List[str]:
    """
    Validates that the provided image paths exist and are accessible.

    Args:
        image_paths (List[str]): List of image file paths.

    Returns:
        List[str]: A filtered list of valid image paths.

    Raises:
        ValueError: If no valid image paths are found.
    """
    valid_paths = [path for path in image_paths if os.path.exists(path)]

    if not valid_paths:
        raise ValueError("None of the provided image paths are valid or accessible.")

    print(f"Validated {len(valid_paths)} image paths.")
    return valid_paths


if __name__ == "__main__":
    # Example usage
    folder = "path_to_images"
    try:
        images = upload_images(folder)
        for img in images:
            print(img)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
