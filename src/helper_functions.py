import os
import pandas as pd
from typing import  List
from pathlib import Path

def csv_checkpoint(csv_path: str, checkpoint_dir: str) -> str:
    """
    Ensures a checkpoint for the given CSV exists.

    Args:
        csv_path (str): Path to the input CSV file.
        checkpoint_dir (str): Directory to save the checkpoint.

    Returns:
        str: Path to the checkpointed CSV file.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    csv_name = os.path.basename(csv_path)
    checkpoint_path = os.path.join(checkpoint_dir, csv_name)

    if not os.path.exists(checkpoint_path):
        df = pd.read_csv(csv_path)
        df.to_csv(checkpoint_path, index=False)
        print(f"Checkpoint created: {checkpoint_path}")
    else:
        print(f"Checkpoint already exists: {checkpoint_path}")

    return checkpoint_path

def validate_generated_csv(csv_path: str, required_columns: List[str]) -> bool:
    """
    Validates that a generated CSV contains the required columns.

    Args:
        csv_path (str): Path to the CSV file.
        required_columns (List[str]): List of required column names.

    Returns:
        bool: True if the CSV is valid, False otherwise.
    """
    if not os.path.exists(csv_path):
        print(f"CSV file does not exist: {csv_path}")
        return False

    df = pd.read_csv(csv_path)
    for column in required_columns:
        if column not in df.columns:
            print(f"Missing required column: {column}")
            return False

    print(f"CSV validation successful: {csv_path}")
    return True

def ensure_directory_exists(directory: str) -> None:
    """
    Ensures the specified directory exists.

    Args:
        directory (str): Path to the directory.
    """
    os.makedirs(directory, exist_ok=True)
    print(f"Directory ensured: {directory}")

def image_check(image_path: str) -> bool:
    """
    Checks if the image file exists and is valid.

    Args:
        image_path (str): Path to the image file.

    Returns:
        bool: True if the image exists and is valid, False otherwise.
    """
    if not os.path.exists(image_path):
        print(f"Image file does not exist: {image_path}")
        return False

    from PIL import Image
    try:
        with Image.open(image_path) as img:
            img.verify()  # Verify that it's an actual image file
        print(f"Image is valid: {image_path}")
        return True
    except Exception as e:
        print(f"Image validation failed for {image_path}: {e}")
        return False

def merge_csv_files(csv_dir: str, output_path: str) -> None:
    """
    Merges all CSV files in a directory into a single CSV file.

    Args:
        csv_dir (str): Directory containing CSV files to merge.
        output_path (str): Path to save the merged CSV file.
    """
    csv_dir_path = Path(csv_dir)
    csv_files = list(csv_dir_path.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in directory: {csv_dir}")
        return

    combined_df = pd.concat((pd.read_csv(csv) for csv in csv_files), ignore_index=True)
    combined_df.to_csv(output_path, index=False)
    print(f"Merged CSV saved to: {output_path}")

def check_missing_embeddings(embeddings_dir: str, csv_path: str, id_column: str) -> List[str]:
    """
    Checks for missing embeddings based on IDs in a CSV file.

    Args:
        embeddings_dir (str): Directory containing embedding files.
        csv_path (str): Path to the CSV file with IDs.
        id_column (str): Column name in the CSV containing IDs.

    Returns:
        List[str]: List of IDs missing embeddings.
    """
    df = pd.read_csv(csv_path)
    ids = df[id_column].unique()
    missing_ids = []

    for id_value in ids:
        embedding_path = os.path.join(embeddings_dir, f"{id_value}.pt")
        if not os.path.exists(embedding_path):
            missing_ids.append(id_value)

    if missing_ids:
        print(f"Missing embeddings for IDs: {missing_ids}")
    else:
        print("All embeddings are accounted for.")

    return missing_ids
