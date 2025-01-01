# **Movie Character Identification with Personalized Generative Models**

This repository implements a framework for movie character identification, with a primary objective of analyzing and comparing generative and discriminative models to understand their respective advantages. Specifically, the project utilizes diffusion-based generative models and CLIP-based discriminative models to classify and identify characters in challenging visual environments.

The framework includes tools for:
- Preparing datasets through automated video frame extraction and organization.
- Training generative and discriminative models tailored for character identification tasks.
- Evaluating performance across various metrics.

This work aims to advance research in character recognition by addressing complex scenarios such as filtered or occluded scenes in movie datasets, while providing insights into the strengths and limitations of both model types.

---

## **Features**

1. **Dataset Preparation**:
   - Automated video frame extraction from specified time intervals.
   - Categorization of frames based on customizable attributes (e.g., characters, scenes).

2. **Model Training**:
   - **Generative Models**: Diffusion-based approaches for generating and identifying character embeddings.
   - **Discriminative Models**: CLIP-based approaches for embedding and classification tasks.

3. **Evaluation and Analysis**:
   - Metrics for assessing accuracy, precision, recall, and other performance indicators.
   - Visual tools for comparing model outputs.

4. **Research Applications**:
   - Designed for complex scenarios, including filtered or occluded scenes in movie datasets.
   - Facilitates experimentation with character identification models.

---

## **Getting Started**

### **1. Clone the Repository**
```bash
git clone https://github.com/your-username/Movie-Character-Identification-With-Personalized-Generative-Models.git
cd Movie-Character-Identification-With-Personalized-Generative-Models
```
### **2. Install Dependencies**

#### **For Local Setup**
1. Create and activate a virtual environment:
   - **Windows**:
     ```bash
     python -m venv env
     .\env\Scripts\activate
     ```
   - **macOS/Linux**:
     ```bash
     python3 -m venv env
     source env/bin/activate
     ```

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
#### **For Colab**
If you are running this project on Google Colab, ensure the following dependencies are installed by adding this to the first cell in your notebook:
```python
!pip install -q torch torchvision torchaudio
!pip install -q git+https://github.com/huggingface/diffusers.git
!pip install -q accelerate transformers pandas matplotlib opencv-python Pillow
!pip install -q git+https://github.com/openai/CLIP.git
```

---

## **Dataset Preparation**

### **1. Frame Extraction**

The `frame_extractor.py` script extracts frames from a video based on time ranges provided in a CSV file. The script takes a **CSV file** and a **movie file** as inputs and saves the extracted frames into folders based on the specified category.

#### **How It Works**
The script reads a CSV file with the following columns:
- `start_time`: Start time in `HH:MM:SS` format.
- `end_time`: End time in `HH:MM:SS` format.
- `category`: The folder name for extracted frames (e.g., character names or scene types).

The video file is processed to extract frames within the specified time ranges, and these frames are organized into subfolders named after the values in the `category` column.


#### **Required Inputs**
1. **CSV File**:
   The CSV file should specify time ranges and categories for extraction. It must contain the following columns:
   - `start_time`: Start time in `HH:MM:SS` format.
   - `end_time`: End time in `HH:MM:SS` format.
   - `category`: The folder name for extracted frames.

   Example (`dataset.csv`):
   ```csv
   characters,start_time,end_time
   CharacterA,00:01:00,00:01:10
   CharacterB,00:02:30,00:02:40
2. **Movie File**:
   The video file from which frames will be extracted. Supported formats include .mp4, .avi, etc.
#### **Running the Script**
To run the script, use the following command:
```bash
python handlers/FrameExtractor.py --csv dataset.csv --video movie.mp4 --output frames/ --category characters --show
```
#### **Arguments**
- `--csv`: Path to the CSV file containing start and end times and category names.
- `--video`: Path to the video file (e.g., `movie.mp4`).
- `--output`: Directory where the extracted frames will be saved.
- `--category`: The column in the CSV to use as the folder name (e.g., `characters`).
- `--show`: *(Optional)* Displays each frame during extraction for verification.

#### **Output**:
The script creates a folder structure like this:
```
frames/
├── CharacterA/
│   ├── CharacterA_frame_0001.jpg
│   ├── CharacterA_frame_0002.jpg
│   └── ...
├── CharacterB/
│   ├── CharacterB_frame_0001.jpg
│   ├── CharacterB_frame_0002.jpg
│   └── ...
```

---
## **Model Training**
This repository supports two models for character identification: Diffusion-based generative models and CLIP-based discriminative models. Below are the steps for training and usage.

---
### **Diffusion-Based Models**

#### **1. Weight Fine-Tuning and Image-Text Embedding Generation**

Generate image-text embeddings and weight parameters for finetuned model for the Diffusion Identifier. 
##### **Usage**

- **Using Colab**

Use the `create_image_text_embedding.ipynb` notebook.

- **Using Local Script**

Run the `imagic_train.py` script locally:
```bash
python handlers/ImagicTrain.py --input_folder <path_to_images> --text_data <path_to_text_file> --output_folder <output_directory>
```
**Arguments:**

- `--input_folder`: Directory containing input images.
- `--text_data`: Path to the text file with descriptions.
- `--output_folder`: Directory to save embeddings.

### **2. Diffusion Identification Model**

Train a model to classify characters using embeddings generated in the previous step.

#### **Usage**

- **Using Colab**
  
Use the `IdentifierModels.ipynb` notebook.

2. Using Local Script:

```bash
Copy code
python models/Diffusion/DiffusionIdentifier.py --embeddings <path_to_embeddings> --labels <path_to_labels> --output_model <output_directory>
   ```

**Arguments:**

--embeddings: Path to embeddings.
--labels: Labels for the embeddings.
--output_model: Directory for the trained model.
