from PIL import Image
import numpy as np
import os

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

def generate_image_path(image_path, embeds_name, alpha, guidance_scale):

    print(f"path entered {image_path}")
    image_name = f"{embeds_name}*alpha:{alpha}^GS:{guidance_scale}.jpg"
    embeds_category = embeds_name.rsplit("_", 1)[0]
    category_folder = os.path.join(image_path, embeds_category)
    os.makedirs(category_folder, exist_ok=True)
    item_path = os.path.join(category_folder, image_name)
    return item_path, image_name

def is_image(file_path):

    try:
        with Image.open(file_path) as img:
            img.verify()  # Verify the image integrity
            return True
    except (IOError, SyntaxError) as e:
        print(f"File {file_path} is not an image or cannot be opened. Error: {e}")
        return False
