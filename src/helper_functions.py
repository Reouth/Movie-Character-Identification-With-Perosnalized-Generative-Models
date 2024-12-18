import pandas as pd
from src import data_upload
import os
import shutil
from pathlib import Path


def move_csv_files(source_dir, destination_dir):
    """
    Move all CSV files from a source directory (and its subdirectories) to a destination directory,
    maintaining the folder structure within the destination directory.

    Args:
        source_dir (str): Path to the source directory containing CSV files.
        destination_dir (str): Path to the destination directory where CSV files will be moved.
    """
    # Check if source directory exists
    if not os.path.exists(source_dir):
        print(f"The source directory '{source_dir}' does not exist.")
        return

    # Create destination directory if it doesn't exist
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    def process_directory(current_dir, folder_name_prefix):
        """
        Recursively process directories and move CSV files to the destination directory.

        Args:
            current_dir (str): Current directory being processed.
            folder_name_prefix (str): Prefix for maintaining folder structure in destination.
        """
        for item in os.listdir(current_dir):
            item_path = os.path.join(current_dir, item)

            if os.path.isdir(item_path):
                # Recurse into subdirectories with updated prefix
                new_prefix = f"{folder_name_prefix}_{item}" if folder_name_prefix else item
                process_directory(item_path, new_prefix)
            elif item.endswith('.csv'):
                # Move CSV files while preserving the folder structure
                destination_folder = os.path.join(destination_dir, folder_name_prefix)
                if not os.path.exists(destination_folder):
                    os.makedirs(destination_folder)
                shutil.move(item_path, os.path.join(destination_folder, item))
                print(f"Moved: {item} to {destination_folder}")

    # Start processing from the source directory
    prefix = Path(source_dir).name
    process_directory(source_dir, prefix)

    print("Operation completed.")


def save_to_csv(SD_loss, df_sd, image_name, csv_file_path):
    """
    Save similarity scores and image details to a CSV file.

    Args:
        SD_loss (dict): Dictionary of similarity scores with keys as embeddings.
        df_sd (pd.DataFrame): Existing DataFrame to append new data to.
        image_name (str): Ground truth image name to associate with the scores.
        csv_file_path (str): Path to save the updated CSV file.
    """
    # Sort similarity scores by value
    sorted_SD = sorted(SD_loss.items(), key=lambda kv: kv[1])

    # Create a DataFrame for the current image's data
    df_image = pd.DataFrame(sorted_SD, columns=['input_SD_embeds', 'SD_loss'])
    df_image.insert(0, 'GT Image name', image_name)

    print(df_image)

    # Append new data to the existing DataFrame
    df_sd = pd.concat([df_sd, df_image], ignore_index=False)

    # Save the combined DataFrame to a CSV file
    df_sd.to_csv(csv_file_path)


def list_csv_files_in_directory(directory_path):
    """
    List all CSV files in the specified directory.

    Args:
        directory_path (str): Path to the directory to scan for CSV files.

    Returns:
        list: List of CSV file names in the directory.
    """
    try:
        items = os.listdir(directory_path)
        csv_files = [item for item in items if
                     item.endswith('.csv') and os.path.isfile(os.path.join(directory_path, item))]
        return csv_files
    except FileNotFoundError:
        print(f"The directory {directory_path} does not exist.")
        return []


def csv_checkpoint(csv_folder, cls, test_image, input_embeds_value=None):
    """
    Check if a test image or embedding exists in a CSV file and return the DataFrame if it exists.

    Args:
        csv_folder (str): Path to the folder containing CSV files.
        cls (str): Class name used to construct the CSV file name.
        test_image (str): Test image name to search for.
        input_embeds_value (optional, str): Specific embedding value to match.

    Returns:
        tuple: (image_flag (bool), df_sd (pd.DataFrame), csv_path (str))
    """
    image_flag = False
    csv_list = list_csv_files_in_directory(csv_folder)
    filepath = f"{cls}_results.csv"
    csv_path = os.path.join(csv_folder, filepath)

    if filepath in csv_list:
        # Load the CSV file if it exists
        df_sd = pd.read_csv(csv_path)
        df_sd = df_sd.drop(columns=['Unnamed: 0'], errors='ignore')
        df_sd.columns = [col.strip() for col in df_sd.columns]

        if 'GT Image name' in df_sd.columns:
            if input_embeds_value is None:
                matches = df_sd[df_sd['GT Image name'] == test_image]
                if not matches.empty:
                    print(f"test_image: {test_image} found in 'GT Image name' column.")
                    image_flag = True
            elif 'input_SD_embeds' in df_sd.columns:
                matches = df_sd[
                    (df_sd['GT Image name'] == test_image) & (df_sd['input_SD_embeds'] == input_embeds_value)]
                if not matches.empty:
                    print(f"test_image: {test_image} and {input_embeds_value} found in the same row.")
                    image_flag = True
    else:
        # Return an empty DataFrame if the file doesn't exist
        df_sd = pd.DataFrame()

    return image_flag, df_sd, csv_path


def generated_image_checkpoint(image_path, embeds_name, alpha, guidance_scale):
    """
    Check if a generated image exists, and if not, prepare for its generation.

    Args:
        image_path (str): Path to the directory where the image is expected to be found.
        embeds_name (str): Name of the embedding used.
        alpha (float): Alpha parameter for generation.
        guidance_scale (float): Guidance scale parameter for generation.

    Returns:
        tuple: (flag (bool), item_path (str), image_name (str))
    """
    print(f"path entered {image_path}")
    image_name = f"{embeds_name}*alpha:{alpha}^GS:{guidance_scale}.jpg"
    embeds_category = embeds_name.rsplit("_", 1)[0]
    category_folder = os.path.join(image_path, embeds_category)
    os.makedirs(category_folder, exist_ok=True)
    flag, item_path = image_check(category_folder, image_name)
    return flag, item_path, image_name


def image_check(base_path, image_name):
    """
    Check if an image exists in a specified directory and verify if it is a valid image.

    Args:
        base_path (str): Path to the directory containing the image.
        image_name (str): Name of the image file to check.

    Returns:
        tuple: (flag (bool), item_path (str))
    """
    item_path = os.path.join(base_path, image_name)
    flag = False

    if os.path.exists(item_path):
        print(f"File exists: {item_path}")
        if data_upload.is_image(item_path):
            flag = True
            print(f"File is an image: {item_path}")
        else:
            print(f"File is not an image: {item_path}")
    else:
        print(f"File does not exist: {item_path}")

    return flag, item_path
