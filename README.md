# Medical Image Segmentation and Tumor Detection

This project implements a complete pipeline for segmenting and detecting nuclei in medical images, specifically focusing on tumor detection and counting.

## Project Structure

```
Project 3 - Segmentation in Medical Imaging/
├── Data/
│   ├── csv/          # Annotation coordinates
│   ├── mask/         # Binary masks for segmentation
│   └── rgb/          # Input RGB images
├── src/
│   ├── main_pipeline.py     # Complete pipeline runner
│   ├── data_pipeline.py     # Dataset pipeline and visualization
│   ├── segmentation.py      # Unsupervised and supervised segmentation
│   ├── detection.py         # Tumor detection and counting
│   ├── bonus_deep_learning.py  # Deep learning bonus section
│   └── utils.py            # Utility functions
├── results/                  # Output results and visualizations
├── splits/                   # Train/validation splits
├── requirements.txt          # Python dependencies
└── README.md               # This file
```

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Complete Pipeline (Recommended)

Run the entire project pipeline at once:

```bash
python3 src/main_pipeline.py
```

This will execute all sections in order:
1. **Dataset Pipeline** - Data preparation and visualization
2. **Segmentation** - Unsupervised and supervised image segmentation
3. **Tumor Detection** - Feature extraction and classification
4. **Bonus Deep Learning** - Deep learning-based segmentation

### Individual Modules

You can also run each module individually:

```bash
# Data Pipeline
python3 src/data_pipeline.py

# Segmentation
python3 src/segmentation.py

# Tumor Detection
python3 src/detection.py

# Bonus Deep Learning
python3 src/bonus_deep_learning.py
```

## Features

- **Dataset Pipeline**: Visualization of images, masks, and bounding boxes
- **Unsupervised Segmentation**: K-means clustering and other methods
- **Supervised Segmentation**: Random Forest and SVM classifiers
- **Tumor Detection**: Feature extraction and SVM-based classification
- **Performance Metrics**: Accuracy, DICE coefficient, Precision, Recall, RMSE

## Dataset Information

The dataset contains:
- RGB images at 0.2 microns-per-pixel resolution
- CSV annotations with bounding box coordinates
- Binary masks for ground truth segmentation
- Multiple samples per patient (identified by TCGA IDs)

## Key Results

- Optimized K-means clustering for image segmentation
- Comparison of RGB vs grayscale performance
- Supervised vs unsupervised segmentation analysis
- Tumor detection with feature-based classification
- Tumor counting with RMSE evaluation

## Citation

This project implements medical image segmentation and tumor detection techniques using computer vision and machine learning approaches. 