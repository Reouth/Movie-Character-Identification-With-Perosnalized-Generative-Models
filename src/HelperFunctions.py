import pandas as pd
from src import DataUpload
import os
import shutil
from pathlib import Path


def move_csv_files(source_dir, destination_dir):

    if not os.path.exists(source_dir):
        print(f"The source directory '{source_dir}' does not exist.")
        return
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    def process_directory(current_dir, folder_name_prefix):

        for item in os.listdir(current_dir):
            item_path = os.path.join(current_dir, item)

            if os.path.isdir(item_path):
                new_prefix = f"{folder_name_prefix}_{item}" if folder_name_prefix else item
                process_directory(item_path, new_prefix)
            elif item.endswith('.csv'):
                destination_folder = os.path.join(destination_dir, folder_name_prefix)
                if not os.path.exists(destination_folder):
                    os.makedirs(destination_folder)
                shutil.move(item_path, os.path.join(str(destination_folder), item))
                print(f"Moved: {item} to {destination_folder}")

    prefix = Path(source_dir).name
    process_directory(source_dir, prefix)

    print("Operation completed.")


def save_to_csv(loss, df, image_name, csv_file_path):
    """
    Save similarity scores and image details to a CSV file.

    Args:
        loss (dict): Dictionary of similarity scores with keys as embeddings.
        df (pd.DataFrame): Existing DataFrame to append new data to.
        image_name (str): Ground truth image name to associate with the scores.
        csv_file_path (str): Path to save the updated CSV file.
    """
    # Sort similarity scores by value
    sorted_df = sorted(loss.items(), key=lambda kv: kv[1])

    # Create a DataFrame for the current image's data
    df_image = pd.DataFrame(sorted_df, columns=['input_embeds', 'loss'])
    df_image.insert(0, 'GT Image name', image_name)

    df = pd.concat([df, df_image], ignore_index=False)
    df.to_csv(csv_file_path)


def list_csv_files_in_directory(directory_path):

    try:
        items = os.listdir(directory_path)
        csv_files = [item for item in items if
                     item.endswith('.csv') and os.path.isfile(os.path.join(directory_path, item))]
        return csv_files
    except FileNotFoundError:
        print(f"The directory {directory_path} does not exist.")
        return []

def file_exists(csv_folder,filepath):
    csv_list = list_csv_files_in_directory(csv_folder)
    if filepath in csv_list:
        return True
    else:
        return False

def get_current_csv(csv_folder, cls):
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
    filename = f"{cls}_results.csv"
    csv_path = os.path.join(csv_folder, filename)

    if file_exists(csv_folder, filename):
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame()
    return df, csv_path

def row_exist(df, test_image, input_embeds_value=None):

    image_flag = False
    df= df.drop(columns=['Unnamed: 0'], errors='ignore')
    df.columns = [col.strip() for col in df.columns]

    if 'GT Image name' in df.columns:
        if input_embeds_value is None:
            matches = df[df['GT Image name'] == test_image]
            if not matches.empty:
                print(f"test_image: {test_image} found in 'GT Image name' column.")
                image_flag = True
        elif 'input_embeds' in df.columns:
            matches = df[
                (df['GT Image name'] == test_image) & (df['input_embeds'] == input_embeds_value)]
            if not matches.empty:
                print(f"test_image: {test_image} and {input_embeds_value} found in the same row.")
                image_flag = True

    return image_flag


def generate_image_path(image_path, embeds_name, alpha, guidance_scale):

    print(f"path entered {image_path}")
    image_name = f"{embeds_name}*alpha:{alpha}^GS:{guidance_scale}.jpg"
    embeds_category = embeds_name.rsplit("_", 1)[0]
    category_folder = os.path.join(image_path, embeds_category)
    os.makedirs(category_folder, exist_ok=True)
    item_path = os.path.join(category_folder, image_name)
    return item_path, image_name


def image_check(item_path):

    flag = False
    if os.path.exists(item_path):
        print(f"File exists: {item_path}")
        if DataUpload.is_image(item_path):
            flag = True
            print(f"File is an image: {item_path}")
        else:
            print(f"File is not an image: {item_path}")
    else:
        print(f"File does not exist: {item_path}")

    return flag
