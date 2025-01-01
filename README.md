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
   ```
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
python preprocessing/FrameExtractor.py --csv dataset.csv --video movie.mp4 --output frames/ --category characters --show
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

## **Diffusion-Based Models**

Diffusion-based generative models for character identification. Below are the steps for training and usage.

### **1. Weight Fine-Tuning and Image-Text Embedding Generation**

Generate image-text embeddings and weight parameters for finetuned model for the Diffusion Identifier. 

#### **Usage**

- **Using Colab**

Use the `CreateImageTextEmbedding.ipynb` notebook.

- **Using Local Script**

Run the `ImagicTrain.py` script locally:

```bash
python ImagicTrain.py \
    --pretrained_model_name_or_path <path_to_pretrained_model> \
    --input_image <path_to_input_image> \
    --target_text <text_describing_output> \
    --output_dir <path_to_output_directory> \
    --resolution 512 \
    --center_crop \
    --train_batch_size 4 \
    --emb_train_steps 500 \
    --max_train_steps 1000 \
    --gradient_accumulation_steps 1 \
    --emb_learning_rate 0.001 \
    --learning_rate 0.000001 \
    --use_8bit_adam \
    --scale_lr \
    --seed 42 \
    --mixed_precision fp16 \
    --push_to_hub \
    --hub_model_id <hub_model_name>
```

### **2. Diffusion Identification Model**

Train a model to classify characters using embeddings generated in the previous step.

#### **Usage**

- **Using Colab**
  
Use the `IdentifierModels.ipynb` notebook.

- **Using Local Script**
- 
Run the `DiffusionIdentifier.py` script locally:

```bash
python DiffusionIdentifier.py --imagic_pretrained_path <path_to_embeddings> \
    --csv_folder <path_to_csv_output> \
    --sd_model_name <stable_diffusion_model> \
    --clip_model_name <clip_model_name> \
    --image_list <path_to_image_list> \
    --category_class \
    --imagic_pipe \
    --alpha 0.5 \
    --seed 42 \
    --height 512 \
    --width 512 \
    --resolution 512 \
    --num_inference_steps 50 
```

### **3. Diffusion Generating Images Model**

Generate images using the trained diffusion model.

#### **Usage**

- **Using Colab**
  
Use the DiffusionImageGenerator.ipynb notebook.

- **Using Local Script**
  
Run the `DiffusionGenerator.py` script locally:


```bash
python DiffusionGenerator.py --input_files <path_to_input_files> \
    --output_folder <path_to_output_folder> \
    --imagic_pretrained_path <path_to_model> \
    --imagic_pipe \
    --sd_model_name <stable_diffusion_model> \
    --clip_model_name <clip_model_name> \
    --seed_range 0 10 \
    --alpha_range 0.0 1.0 \
    --guidance_scale_range 7.0 8.0 \
    --height 512 \
    --width 512 \
    --num_inference_steps 50
   ```

---

## **CLIP-Based Model**
CLIP-based discriminative model for character identification. Below are the steps for training and usage.

### **1. Image Identification Model**

Classify characters from real or generated images (eg. Diffusion Generator images) using CLIP.

#### **Usage**

- **Using Colab**

Use the `IdentifierModels.ipynb` notebook.

- **Using Local Script**
  
Run the `CLIPIdentifier.py` script locally:

```bash
python CLIPIdentifier.py --input_dir <path_to_images> \
    --output_dir <path_to_csv_results> \
    --image_list <path_to_image_list> \
    --model_name ViT-B/32 \
    --device cuda
```
---

## **Evaluation**
Evaluate the performance of models using metrics such as Top-K Accuracy and Mean Average Precision (mAP). 
These metrics assess the ability of embeddings to classify characters correctly and rank relevant results.

### **1. Top-K Accuracy**
Calculate Top-K accuracy to evaluate how often the correct class appears in the top K predictions.

#### **Usage**

- **Using Colab**

Use the `SimilarityCompare.ipynb` notebook.

- **Using Local Script**
  
Run the MetricsCalc.py script with the topk subcommand locally:

``` bash
python MetricsCalc.py topk \
    --input_folder <path_to_csv_results> \
    --output_folder <path_to_output_results> \
    --k_range 5 \
    --clip_csv \
    --avg \
    --pred_column loss
```
- **Note:**
--clip_csv: (Optional) Use this flag for CLIP model results (default assumes SD results).
--avg: (Optional) Use this flag to calculate average accuracy over all embeddings.
--pred_column: (Optional) Column used for predictions (e.g., loss for SD or scores for CLIP). Default is loss.

- **Outputs**
For each value of K in --k_range, a CSV file is created in the --output_folder:
Contains Top-K accuracy for each class.
Includes overall Top-K accuracy across all classes.

## **2. Mean Average Precision (mAP)**
Calculate mAP to evaluate how well the model ranks relevant result.

#### **Usage**

- **Using Colab**

Use the `SimilarityCompare.ipynb` notebook.

- **Using Local Script**

Run the MetricsCalc.py script with the map subcommand locally:

```bash
python MetricsCalc.py map \
    --input_folder <path_to_csv_results> \
    --output_folder <path_to_output_results> \
    --clip_csv
```

- **Note:**
--clip_csv: (Optional) Use this flag for CLIP model results (default assumes SD results).

- **Outputs**
A CSV file named average_precision_results.csv is created in the --output_folder:
Contains Average Precision (AP) for each query.
Includes Mean Average Precision (mAP) across all queries.

