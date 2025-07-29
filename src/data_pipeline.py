"""
Data Pipeline for Medical Image Segmentation Project
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from utils import (
    extract_patient_id, load_image_and_mask, load_annotations,
    create_bounding_boxes, draw_bounding_boxes, visualize_segmentation
)


class DataPipeline:
    def __init__(self, data_dir="Data"):
        self.data_dir = data_dir
        self.csv_dir = os.path.join(data_dir, "csv")
        self.rgb_dir = os.path.join(data_dir, "rgb")
        self.mask_dir = os.path.join(data_dir, "mask")
        
        # Create output directories
        os.makedirs("results", exist_ok=True)
        os.makedirs("splits", exist_ok=True)
        
    def get_unique_subjects(self):
        """
        Extract unique subjects from filenames.
        """
        subjects = set()
        
        # Get all CSV files
        csv_files = [f for f in os.listdir(self.csv_dir) if f.endswith('.csv')]
        
        for csv_file in csv_files:
            patient_id = extract_patient_id(csv_file)
            if patient_id:
                subjects.add(patient_id)
        
        return list(subjects)
    
    def get_files_for_subject(self, subject_id):
        """
        Get all files (csv, rgb, mask) for a given subject.
        """
        files = []
        
        # Get all CSV files for this subject
        csv_files = [f for f in os.listdir(self.csv_dir) 
                    if f.startswith(subject_id) and f.endswith('.csv')]
        
        for csv_file in csv_files:
            # Extract the unique identifier part (everything after the subject_id)
            # Format: TCGA-S3-AA15-DX1_id-5ea40a6addda5f839898f24a_left-57268_top-29680_bottom-29958_right-57547.csv
            # We want: id-5ea40a6addda5f839898f24a_left-57268_top-29680_bottom-29958_right-57547
            identifier = csv_file[len(subject_id)+1:-4]  # Remove subject_id, underscore, and .csv extension
            
            # Find corresponding RGB and mask files
            rgb_file = f"{subject_id}_{identifier}.png"
            mask_file = f"{subject_id}_{identifier}.png"
            
            if (os.path.exists(os.path.join(self.rgb_dir, rgb_file)) and 
                os.path.exists(os.path.join(self.mask_dir, mask_file))):
                    files.append({
                        'csv': csv_file,
                        'rgb': rgb_file,
                        'mask': mask_file,
                        'subject_id': subject_id
                    })
        
        return files
    
    def visualize_sample_subject(self, subject_id=None):
        """
        Visualize three images for a sample subject.
        """
        if subject_id is None:
            subjects = self.get_unique_subjects()
            subject_id = subjects[0]  # Use first subject as sample
        
        files = self.get_files_for_subject(subject_id)
        if not files:
            print(f"No files found for subject {subject_id}")
            return
        
        # Use the first file for visualization
        sample_file = files[0]
        
        # Load data
        csv_path = os.path.join(self.csv_dir, sample_file['csv'])
        rgb_path = os.path.join(self.rgb_dir, sample_file['rgb'])
        mask_path = os.path.join(self.mask_dir, sample_file['mask'])
        
        image, mask = load_image_and_mask(rgb_path, mask_path)
        annotations = load_annotations(csv_path)
        
        # Create bounding boxes
        boxes = create_bounding_boxes(annotations)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # Mask
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title("Segmentation Mask")
        axes[1].axis('off')
        
        # Image with bounding boxes
        img_with_boxes = draw_bounding_boxes(image, boxes)
        axes[2].imshow(img_with_boxes)
        axes[2].set_title("Image with Nuclei Labels")
        axes[2].axis('off')
        
        plt.suptitle(f"Sample Visualization for Subject: {subject_id}")
        plt.tight_layout()
        plt.savefig("results/sample_visualization.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Visualization saved for subject: {subject_id}")
        print(f"Number of annotations: {len(boxes)}")
        
        return {
            'subject_id': subject_id,
            'image': image,
            'mask': mask,
            'annotations': annotations,
            'boxes': boxes
        }
    
    def create_train_val_splits(self):
        """
        Create training and validation splits (80% train, 20% val).
        """
        subjects = self.get_unique_subjects()
        
        # Shuffle subjects
        np.random.seed(42)  # For reproducibility
        np.random.shuffle(subjects)
        
        # Split into train and validation
        split_idx = int(0.8 * len(subjects))
        train_subjects = subjects[:split_idx]
        val_subjects = subjects[split_idx:]
        
        # Create DataFrames
        train_df = pd.DataFrame({'subject_id': train_subjects})
        val_df = pd.DataFrame({'subject_id': val_subjects})
        
        # Save splits
        train_df.to_csv("splits/train_subjects.csv", index=False)
        val_df.to_csv("splits/validate_subjects.csv", index=False)
        
        print(f"Created train/validation splits:")
        print(f"Training subjects: {len(train_subjects)}")
        print(f"Validation subjects: {len(val_subjects)}")
        print(f"Total subjects: {len(subjects)}")
        
        return train_subjects, val_subjects
    
    def get_all_files_for_split(self, subjects):
        """
        Get all files for a given split of subjects.
        """
        all_files = []
        
        for subject_id in subjects:
            files = self.get_files_for_subject(subject_id)
            all_files.extend(files)
        
        return all_files
    
    def analyze_dataset(self):
        """
        Analyze the dataset statistics.
        """
        subjects = self.get_unique_subjects()
        
        print("Dataset Analysis:")
        print("=" * 50)
        print(f"Total unique subjects: {len(subjects)}")
        
        # Count files per subject
        files_per_subject = []
        total_files = 0
        
        for subject_id in subjects:
            files = self.get_files_for_subject(subject_id)
            files_per_subject.append(len(files))
            total_files += len(files)
        
        print(f"Total files: {total_files}")
        print(f"Average files per subject: {np.mean(files_per_subject):.2f}")
        print(f"Min files per subject: {min(files_per_subject)}")
        print(f"Max files per subject: {max(files_per_subject)}")
        
        # Analyze annotations
        annotation_counts = []
        tumor_counts = []
        
        for subject_id in subjects[:10]:  # Sample first 10 subjects
            files = self.get_files_for_subject(subject_id)
            for file_info in files[:3]:  # Sample first 3 files per subject
                csv_path = os.path.join(self.csv_dir, file_info['csv'])
                annotations = load_annotations(csv_path)
                
                annotation_counts.append(len(annotations))
                
                # Count tumors
                tumor_count = len(annotations[annotations['raw_classification'] == 'tumor'])
                tumor_counts.append(tumor_count)
        
        print(f"Average annotations per file: {np.mean(annotation_counts):.2f}")
        print(f"Average tumor annotations per file: {np.mean(tumor_counts):.2f}")
        
        return {
            'total_subjects': len(subjects),
            'total_files': total_files,
            'avg_files_per_subject': np.mean(files_per_subject),
            'avg_annotations_per_file': np.mean(annotation_counts),
            'avg_tumors_per_file': np.mean(tumor_counts)
        }


def main():
    """
    Main function to run the data pipeline.
    """
    print("Medical Image Segmentation - Data Pipeline")
    print("=" * 50)
    
    # Initialize pipeline
    pipeline = DataPipeline()
    
    # Analyze dataset
    print("\n1. Analyzing dataset...")
    stats = pipeline.analyze_dataset()
    
    # Create train/validation splits
    print("\n2. Creating train/validation splits...")
    train_subjects, val_subjects = pipeline.create_train_val_splits()
    
    # Visualize sample subject
    print("\n3. Creating sample visualization...")
    sample_data = pipeline.visualize_sample_subject()
    
    print("\nData pipeline completed successfully!")
    print("Files created:")
    print("- splits/train_subjects.csv")
    print("- splits/validate_subjects.csv")
    print("- results/sample_visualization.png")


if __name__ == "__main__":
    main() 