# COMP9517 2025 Term 1 Group Project - Aerial Scene Classification

## Authors
Group project 9517: Zijia Wang z5561819 Chenxi Dong z5461249 Zhuo Chen z5641208 Ziheng Zhang z5450932
Yuhang Liang z5484376


## Overview
This project is part of the COMP9517 Computer Vision course at UNSW, Term 1, 2025. The goal is to develop and compare computer vision methods for classifying aerial scenes from the "SkyView: An Aerial Landscape Dataset" (15 categories, 800 images per category). The project implements two approaches:

1. **Machine Learning (ML) Pipeline**:
   - Feature extraction: SIFT, LBP, and color histograms.
   - Feature encoding: Bag-of-Words (BoW) for SIFT.
   - Feature fusion: Combining SIFT, LBP, and color histograms with PCA dimensionality reduction.
   - Classification: SVM, Random Forest, KNN, and Decision Tree.

2. **Deep Learning (DL) Pipeline**:
   - Models: ResNet-18 and MobileNetV3-Small (pretrained).
   - Data augmentation: Random flips, crops, color jitter, rotation, and CutMix.
   - Advanced methods: Imbalanced data handling, Score-CAM for explainability, and occlusion robustness testing.
   - Evaluation: Accuracy, F1-score, precision, recall, and IoU.

The project is implemented in Python, using libraries such as PyTorch, scikit-learn, and OpenCV. The code is modularized for clarity and reproducibility, with experiments managed in Jupyter Notebooks.

## Dependencies
To run the project, install the following dependencies:

- **Python**: 3.8 or higher
- **Libraries**:
  ```bash
  pip install torch torchvision numpy pandas scikit-learn opencv-python matplotlib seaborn tqdm pytorch-grad-cam
  ```
- **Hardware**: GPU recommended (e.g., NVIDIA L4 on Google Colab) for deep learning experiments.
- **Dataset**: SkyView: An Aerial Landscape Dataset ([Kaggle link](https://www.kaggle.com/datasets/ankit1743/skvview-an-aerial-landscape-dataset)).

## Data Preparation
1. **Download the Dataset**:
   - Download the SkyView dataset from Kaggle.
   - Extract the dataset to a directory (e.g., `Aerial_Landscapes/`).
   - The dataset should have 15 subdirectories (one per category), each containing 800 images.

2. **Set Data Path**:
   - For the ML pipeline, update the `data_dir` variable in `9517ML.ipynb` to the dataset path (e.g., `Aerial_Landscapes/`).
   - For the DL pipeline, update the `data_dir` variable in `aerial_classification_main.py` or `9517DL.ipynb` to the dataset path.

3. **Directory Structure**:
   ```
   Aerial_Landscapes/
   ├── Railway/
   ├── City/
   ├── Airport/
   ...
   └── River/
   ```

## Installation
1. **Clone the Repository** (if applicable):
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Set Up Google Colab** (optional):
   - Upload the project folder to Google Drive.
   - Open `9517ML.ipynb` or `9517DL.ipynb` in Google Colab.
   - Mount Google Drive and update the data path as shown in the notebooks.

## Running the Code
The project consists of two main pipelines, each with its own entry point.

### Machine Learning Pipeline
1. **Notebook**: `9517ML.ipynb`
2. **Steps**:
   - Open `9517ML.ipynb` in Jupyter Notebook or Google Colab.
   - Update the `data_dir` variable to point to the dataset path.
   - Run all cells sequentially to:
     - Load and preprocess data.
     - Extract SIFT, LBP, and color histogram features.
     - Train a BoW model for SIFT features.
     - Fuse features and apply PCA.
     - Train and evaluate classifiers (SVM, Random Forest, KNN, Decision Tree).
     - Output results in a table.
3. **Expected Runtime**: ~20-30 minutes on Google Colab with NVIDIA L4 GPU.


### Deep Learning Pipeline
1. **Script**: `aerial_classification_main.py`
   - **Command**:
     ```bash
     python src/DL/aerial_classification_main.py
     ```
   - **Steps**:
     - Update the `data_dir` variable in `aerial_classification_main.py` to the dataset path.
     - Run the script to:
       - Load data with two augmentation strategies.
       - Train ResNet-18 and MobileNetV3-Small (with/without CutMix).
       - Evaluate models on normal and occluded data.
       - Generate Score-CAM heatmaps for explainability.
       - Save results to `final_comparison.csv`.
   - **Expected Runtime**: ~30-40 minutes on Google Colab with NVIDIA L4 GPU.

2. **Notebook**: `9517DL.ipynb`
   - Open `9517DL.ipynb` in Jupyter Notebook or Google Colab.
   - Update the data path and run all cells to replicate the experiments in `aerial_classification_main.py`.
   - Additional experiments (e.g., imbalanced data) are included.
   - Outputs a summary table in CSV and Markdown formats.

## Outputs
- **Machine Learning Pipeline**:
  - **Results Table**: Printed in `9517ML.ipynb`, showing accuracy, F1-score, precision, recall, IoU, and train loss for each feature-classifier combination.
  - **Visualizations**: Confusion matrices for each experiment.

- **Deep Learning Pipeline**:
  - **Results File**: `9517DL.ipynb`, containing performance metrics (accuracy, F1, precision, recall, IoU, train loss, training time) for normal, imbalanced, and occlusion test scenarios.
  - **Visualizations**:
    - Confusion matrices for each model and scenario.
    - Score-CAM heatmap for a sample image (printed in `aerial_classification_main.py`).
  - **Summary Table**: Printed in `9517DL.ipynb` in Markdown format.

## Directory Structure
```
project_root/
├── src/
│   ├── ML/
│   │   ├── aerial_data_handler.py
│   │   ├── aerial_sift_extractor.py
│   │   ├── aerial_lbp_generator.py
│   │   ├── aerial_bow_builder.py
│   │   ├── aerial_feature_combiner.py
│   │   ├── aerial_metrics_analyzer.py
│   ├── DL/
│   │   ├── aerial_data_loader.py
│   │   ├── aerial_model_factory.py
│   │   ├── aerial_training_utils.py
│   │   ├── aerial_occlusion_test.py
│   │   ├── aerial_scorecam.py
│   │   ├── aerial_classification_main.py
├── 9517ML.ipynb
├── 9517DL.ipynb
├── README.md
```

## Notes
- **Dataset Size**: The full dataset (12000 images) is used by default. To reduce computation, set `sample_ratio < 1` in `aerial_data_loader.py` (DL) or subsample in `9517ML.ipynb` (ML).
- **Hardware**: GPU acceleration is highly recommended. Google Colab with a free GPU (e.g., NVIDIA L4) is sufficient.
- **Error Handling**: Ensure the dataset path is correct to avoid file loading errors. Check for missing dependencies before running.
- **Submission**: The code submission (ZIP file) excludes the dataset, trained models, and output images to meet the 25MB limit.
 

---
*Last updated: April 25, 2025*