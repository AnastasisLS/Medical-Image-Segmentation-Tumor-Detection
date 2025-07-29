"""
Utility functions for medical image segmentation and tumor detection.
"""

import os
import re
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


def extract_patient_id(filename):
    """
    Extract patient ID from filename.
    Format: TCGA-A1-A0SP-DX1_id-... -> TCGA-A1-A0SP-DX1
    """
    match = re.match(r'(TCGA-[A-Z0-9]+-[A-Z0-9]+-DX\d+)', filename)
    return match.group(1) if match else None


def load_image_and_mask(image_path, mask_path):
    """
    Load image and corresponding mask.
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    return image, mask


def load_annotations(csv_path):
    """
    Load annotations from CSV file.
    """
    df = pd.read_csv(csv_path)
    return df


def calculate_dice_coefficient(y_true, y_pred):
    """
    Calculate DICE coefficient (F1 score) for binary segmentation.
    """
    intersection = np.logical_and(y_true, y_pred).sum()
    union = y_true.sum() + y_pred.sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return 2 * intersection / union


def calculate_metrics(y_true, y_pred):
    """
    Calculate various performance metrics.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    dice = calculate_dice_coefficient(y_true, y_pred)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'dice': dice
    }


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """
    Plot confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    return cm


def visualize_segmentation(image, mask, title="Segmentation Result"):
    """
    Visualize segmentation results.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title("Segmentation Mask")
    axes[1].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    return fig


def extract_features(image, mask=None):
    """
    Extract features from image for classification.
    """
    features = []
    
    # Convert to grayscale if RGB
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Basic statistical features
    features.extend([
        np.mean(gray),
        np.std(gray),
        np.median(gray),
        np.max(gray),
        np.min(gray)
    ])
    
    # Histogram features
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    features.extend([
        np.percentile(hist, 25),
        np.percentile(hist, 50),
        np.percentile(hist, 75),
        np.percentile(hist, 90)
    ])
    
    # Texture features (GLCM-like)
    # Simple gradient features
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    features.extend([
        np.mean(np.abs(grad_x)),
        np.mean(np.abs(grad_y)),
        np.std(grad_x),
        np.std(grad_y)
    ])
    
    # Shape features if mask is provided
    if mask is not None:
        contours, _ = cv2.findContours(mask.astype(np.uint8), 
                                     cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Area and perimeter
            area = cv2.contourArea(contours[0])
            perimeter = cv2.arcLength(contours[0], True)
            features.extend([area, perimeter])
            
            # Circularity
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                features.append(circularity)
            else:
                features.append(0)
        else:
            features.extend([0, 0, 0])
    
    return np.array(features)


def create_bounding_boxes(annotations_df):
    """
    Create bounding boxes from annotations.
    """
    boxes = []
    for _, row in annotations_df.iterrows():
        if pd.notna(row['xmin']) and pd.notna(row['ymin']) and \
           pd.notna(row['xmax']) and pd.notna(row['ymax']):
            box = {
                'xmin': int(row['xmin']),
                'ymin': int(row['ymin']),
                'xmax': int(row['xmax']),
                'ymax': int(row['ymax']),
                'class': row['raw_classification'],
                'type': row['type']
            }
            boxes.append(box)
    return boxes


def draw_bounding_boxes(image, boxes, color=(255, 0, 0), thickness=2):
    """
    Draw bounding boxes on image.
    """
    img_with_boxes = image.copy()
    
    for box in boxes:
        cv2.rectangle(img_with_boxes, 
                     (box['xmin'], box['ymin']), 
                     (box['xmax'], box['ymax']), 
                     color, thickness)
        
        # Add label
        label = box['class']
        cv2.putText(img_with_boxes, label, 
                   (box['xmin'], box['ymin'] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
    
    return img_with_boxes


def save_results(results, filename):
    """
    Save results to file.
    """
    os.makedirs('results', exist_ok=True)
    with open(f'results/{filename}', 'w') as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")


def print_metrics(metrics, title="Performance Metrics"):
    """
    Print formatted metrics.
    """
    print(f"\n{title}")
    print("=" * 50)
    for key, value in metrics.items():
        print(f"{key.capitalize()}: {value:.4f}")
    print("=" * 50) 