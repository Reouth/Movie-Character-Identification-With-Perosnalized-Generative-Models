# **Movie-Character-Identification-With-Perosnalized-Generative-Models**


This repository implements a framework for movie character identification, focusing on leveraging generative and deterministic models to analyze and classify characters in challenging visual environments. The project includes tools for preparing datasets through automated video frame extraction and organization, training models tailored for re-identification tasks, and evaluating their performance on various metrics. This work aims to advance research in character recognition by addressing complex scenarios such as filtered or occluded scenes in movie datasets.

---

## **Features**

### **Dataset Preparation**
- Extract frames from videos using user-defined time intervals.
- Categorize frames based on customizable attributes (e.g., characters, scenes, or metadata from a CSV file).
- Flexible configuration options for dataset creation.

### **Model Training**
- **Generative Models**: Implement diffusion models to capture complex visual characteristics.
- **Deterministic Models**: Leverage CLIP-based approaches for embedding and recognition tasks.

### **Evaluation and Analysis**
- Robust evaluation metrics for character identification and re-identification.
- Tools for visualizing model outputs and comparing generative versus deterministic approaches.

### **Research Focus**
- Supports experimentation with re-identification tasks.
- Designed to address challenges in heavily filtered scenes and complex movie datasets.

---

## **Getting Started**

### **1. Clone the Repository**
```bash
git clone https://github.com/your-username/movie-character-reid.git
cd movie-character-reid
