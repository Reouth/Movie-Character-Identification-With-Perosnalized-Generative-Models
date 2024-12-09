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

---

## **Dataset Preparation**

### **1. Frame Extraction**

The `frame_extractor.py` script is used to extract video frames based on start and end times provided in a CSV file. This script allows you to specify a **character names column** (or any other category column) to organize frames into folders for each category.

#### **How It Works**
The script reads a CSV file containing `start_time`, `end_time`, and a **category column** (e.g., `characters`). It uses these details to extract frames from a video file and save them into category-specific folders.

---

#### **Example CSV File**
The CSV file should contain the following columns:
- A **category column**: Specifies the folder name for the extracted frames (e.g., `characters`).
- `start_time`: The start time for frame extraction in `HH:MM:SS` format.
- `end_time`: The end time for frame extraction in `HH:MM:SS` format.

Example (`dataset.csv`):
```csv
characters,start_time,end_time
CharacterA,00:01:00,00:01:10
CharacterB,00:02:30,00:02:40
