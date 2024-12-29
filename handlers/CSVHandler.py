import pandas as pd
import os
import shutil
from pathlib import Path
import pathlib

def merge_csv_results(base_path,folders_names,output_dir):
    # Get the list of csv files from the first folder (assuming all folders have the same csv files)
    f1_path = os.path.join(base_path, folders_names[0])
    csv_files = [f for f in os.listdir(f1_path) if f.endswith('.csv')]

    # Dictionary to store dataframes for each csv file
    csv_data = {csv_file: [] for csv_file in csv_files}

    # Loop through each folder and each csv file
    for folder in folders_names:
        for csv_file in csv_files:
            file_path = os.path.join(base_path, folder, csv_file)
            df = pd.read_csv(file_path)
            last_row = df.iloc[-1]
            last_row['folder_name'] = folder  # Add the folder name as a new column
            csv_data[csv_file].append(last_row)

    os.makedirs(output_dir, exist_ok=True)

    for csv_file, rows in csv_data.items():
        merged_df = pd.DataFrame(rows)
        # Sort the dataframe by 'Top_k_accuracy' column in descending order
        # sorted_df = merged_df.sort_values(by='Top_k_accuracy', ascending=False)
        output_path = os.path.join(output_dir, csv_file)
        merged_df.to_csv(output_path, index=False)

    print("CSV files have been created with the last rows from each folder.")

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


def save_to_csv(loss, df, image_name, csv_file_path,reverse=False):
    """
    Save similarity scores and image details to a CSV file.

    Args:
        loss (dict): Dictionary of similarity scores with keys as embeddings.
        df (pd.DataFrame): Existing DataFrame to append new data to.
        image_name (str): Ground truth image name to associate with the scores.
        csv_file_path (str): Path to save the updated CSV file.
    """
    # Sort similarity scores by value
    sorted_df = sorted(loss.items(), key=lambda kv: kv[1], reverse=reverse)

    # Create a DataFrame for the current image's data
    df_image = pd.DataFrame(sorted_df, columns=['input_embeds', 'loss'])
    df_image.insert(0, 'GT Image name', image_name)

    df = pd.concat([df, df_image], ignore_index=False)
    df.to_csv(csv_file_path,  index=False)


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

    row_flag = False
    df.columns = [col.strip() for col in df.columns]

    if 'GT Image name' in df.columns:
        if input_embeds_value is None:
            matches = df[df['GT Image name'] == test_image]
            if not matches.empty:
                print(f"test_image: {test_image} found in 'GT Image name' column.")
                row_flag = True
        elif 'input_embeds' in df.columns:
            matches = df[
                (df['GT Image name'] == test_image) & (df['input_embeds'] == input_embeds_value)]
            if not matches.empty:
                print(f"test_image: {test_image} and {input_embeds_value} found in the same row.")
                row_flag = True

    return row_flag

def upload_csvs(csv_dir_path):

    csv_dir = pathlib.Path(csv_dir_path)
    csv_paths = list(csv_dir.glob("*.csv"))  # Collect all CSV files with .csv extension
    return csv_paths