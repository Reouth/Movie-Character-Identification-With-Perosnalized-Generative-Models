
import argparse
import os
import logging
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def convert_time_to_seconds(time_str: str) -> tuple[str, int]:
    """Converts a time string (HH:MM:SS) to total seconds."""
    try:
        time_obj = datetime.strptime(time_str, "%H:%M:%S").time()
        total_seconds = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second
        return time_str, total_seconds
    except ValueError as e:
        logging.error(f"Invalid time format: {time_str}. Expected HH:MM:SS.")
        raise e

def save_frames(output_dir: str, category: str, video_file: str,
                start_time: str, end_time: str, show_frame: bool = False) -> None:
    """Extracts and saves video frames within a time range to category-specific folders."""
    folder_path = os.path.join(output_dir, category.replace(" ", "_"))
    os.makedirs(folder_path, exist_ok=True)

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        logging.error(f"Unable to open video file: {video_file}")
        return

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    logging.info(f"Frame rate: {frame_rate:.2f} fps")

    start_time_formatted, start_time_seconds = convert_time_to_seconds(start_time)
    end_time_formatted, end_time_seconds = convert_time_to_seconds(end_time)
    logging.info(f"Start time: {start_time_formatted}, End time: {end_time_formatted}")

    start_frame = int(start_time_seconds * frame_rate)
    end_frame = int(end_time_seconds * frame_rate)
    logging.info(f"Start frame: {start_frame}, End frame: {end_frame}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    current_frame = start_frame
    while current_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            logging.error(f"Failed to read frame {current_frame}")
            break

        if show_frame:
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.title(f"Frame {current_frame}")
            plt.axis('off')
            plt.show()

        frame_filename = os.path.join(folder_path, f"{category}_frame_{current_frame:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        logging.info(f"Saved frame {current_frame} as {frame_filename}")

        current_frame += 1

    cap.release()
    logging.info(f"Frame extraction completed for category: {category}.")

def process_csv(csv_file: str, video_file: str, output_dir: str, category_column: str, show_frame: bool = False) -> None:
    """Processes the input CSV file and extracts frames for each row."""
    try:
        data = pd.read_csv(csv_file)
        required_columns = {category_column, 'start_time', 'end_time'}
        if not required_columns.issubset(data.columns):
            raise ValueError(f"CSV file must contain columns: {', '.join(required_columns)}")

        for _, row in data.iterrows():
            category = str(row[category_column])
            start_time = row['start_time']
            end_time = row['end_time']
            logging.info(f"Processing category: {category} from {start_time} to {end_time}")
            save_frames(output_dir, category, video_file, start_time, end_time, show_frame)
    except Exception as e:
        logging.error(f"Failed to process CSV file: {csv_file}. Error: {e}")
        raise e

    if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Extract frames from a video based on a CSV file.")
        parser.add_argument("--csv", required=True, help="Path to the CSV file with category and time data.")
        parser.add_argument("--video", required=True, help="Path to the video file.")
        parser.add_argument("--output", required=True, help="Directory to save the extracted frames.")
        parser.add_argument("--category", required=True, help="The column in the CSV to use as the category.")
        parser.add_argument("--show", action="store_true", help="Display frames during extraction.")

        args = parser.parse_args()

        process_csv(args.csv, args.video, args.output, args.category, args.show)
