"""
Tumor Detection Module for Medical Image Segmentation Project
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error
import seaborn as sns
from utils import (
    load_image_and_mask, load_annotations, calculate_metrics,
    print_metrics, save_results, extract_features
)


class TumorDetectionPipeline:
    def __init__(self, data_dir="Data"):
        self.data_dir = data_dir
        self.csv_dir = os.path.join(data_dir, "csv")
        self.rgb_dir = os.path.join(data_dir, "rgb")
        self.mask_dir = os.path.join(data_dir, "mask")
        
        # Load train/validation splits
        self.train_subjects = pd.read_csv("splits/train_subjects.csv")['subject_id'].tolist()
        self.val_subjects = pd.read_csv("splits/validate_subjects.csv")['subject_id'].tolist()
        
        # Create output directories
        os.makedirs("results", exist_ok=True)
        
    def get_files_for_subjects(self, subjects, max_files_per_subject=5):
        """
        Get files for given subjects with limit per subject.
        """
        files = []
        for subject_id in subjects:
            subject_files = self.get_files_for_subject(subject_id)
            files.extend(subject_files[:max_files_per_subject])
        return files
    
    def get_files_for_subject(self, subject_id):
        """
        Get all files for a given subject.
        """
        files = []
        csv_files = [f for f in os.listdir(self.csv_dir) 
                    if f.startswith(subject_id) and f.endswith('.csv')]
        
        for csv_file in csv_files:
            # Extract the identifier correctly
            identifier = csv_file[len(subject_id)+1:-4]  # Remove subject_id, underscore, and .csv extension
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
    
    def prepare_detection_data(self, subjects, use_masked_regions=True):
        """
        Prepare data for tumor detection.
        """
        X = []
        y = []
        image_counts = []  # For counting tumors per image
        
        files = self.get_files_for_subjects(subjects, max_files_per_subject=3)
        
        for file_info in files:
            # Load data
            csv_path = os.path.join(self.csv_dir, file_info['csv'])
            rgb_path = os.path.join(self.rgb_dir, file_info['rgb'])
            mask_path = os.path.join(self.mask_dir, file_info['mask'])
            
            image, mask = load_image_and_mask(rgb_path, mask_path)
            annotations = load_annotations(csv_path)
            
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Ensure same size
            if image.shape[:2] != mask.shape:
                # Resize mask to match image
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
            
            # Create regions of interest
            if use_masked_regions:
                # Use masked regions (multiply image with mask)
                roi = cv2.bitwise_and(gray, gray, mask=mask)
            else:
                # Use original image
                roi = gray
            
            # Remove background noise
            roi_cleaned = self.remove_background_noise(roi)
            
            # Extract features for each annotation
            tumor_count = 0
            non_tumor_count = 0
            
            for _, row in annotations.iterrows():
                if pd.notna(row['xmin']) and pd.notna(row['ymin']) and \
                   pd.notna(row['xmax']) and pd.notna(row['ymax']):
                    
                    xmin, ymin = int(row['xmin']), int(row['ymin'])
                    xmax, ymax = int(row['xmax']), int(row['ymax'])
                    
                    # Ensure coordinates are within bounds
                    h, w = roi_cleaned.shape
                    xmin = max(0, min(xmin, w-1))
                    ymin = max(0, min(ymin, h-1))
                    xmax = max(0, min(xmax, w-1))
                    ymax = max(0, min(ymax, h-1))
                    
                    # Extract region
                    region = roi_cleaned[ymin:ymax, xmin:xmax]
                    
                    if region.size > 0:
                        # Extract features
                        features = self.extract_region_features(region)
                        
                        # Determine label
                        if row['raw_classification'] == 'tumor':
                            label = 1
                            tumor_count += 1
                        else:
                            label = 0
                            non_tumor_count += 1
                        
                        X.append(features)
                        y.append(label)
            
            # Store tumor count for this image
            image_counts.append(tumor_count)
        
        return np.array(X), np.array(y), image_counts
    
    def remove_background_noise(self, image):
        """
        Remove background noise from image.
        """
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        
        # Apply threshold to remove very dark regions
        _, thresh = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to clean up
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        return cleaned
    
    def extract_region_features(self, region):
        """
        Extract comprehensive features from a region.
        """
        features = []
        
        # Basic statistical features
        features.extend([
            np.mean(region),
            np.std(region),
            np.median(region),
            np.max(region),
            np.min(region),
            np.percentile(region, 25),
            np.percentile(region, 75),
            np.percentile(region, 90)
        ])
        
        # Shape features
        features.extend([
            region.shape[0],  # height
            region.shape[1],  # width
            region.shape[0] * region.shape[1],  # area
            region.shape[0] / max(region.shape[1], 1),  # aspect ratio
        ])
        
        # Texture features
        # Gradient features
        grad_x = cv2.Sobel(region, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(region, cv2.CV_64F, 0, 1, ksize=3)
        
        features.extend([
            np.mean(np.abs(grad_x)),
            np.mean(np.abs(grad_y)),
            np.std(grad_x),
            np.std(grad_y),
            np.mean(np.sqrt(grad_x**2 + grad_y**2))  # gradient magnitude
        ])
        
        # Histogram features
        hist = cv2.calcHist([region], [0], None, [256], [0, 256])
        hist = hist.flatten()
        
        features.extend([
            np.argmax(hist),  # peak intensity
            np.sum(hist > np.mean(hist)),  # number of intensity levels above mean
            np.std(hist)
        ])
        
        # Local Binary Pattern-like features
        lbp_features = self.compute_lbp_features(region)
        features.extend(lbp_features)
        
        # Haralick-like texture features
        texture_features = self.compute_texture_features(region)
        features.extend(texture_features)
        
        return np.array(features)
    
    def compute_lbp_features(self, region):
        """
        Compute Local Binary Pattern features.
        """
        # Simplified LBP implementation
        lbp = np.zeros_like(region)
        
        for i in range(1, region.shape[0]-1):
            for j in range(1, region.shape[1]-1):
                center = region[i, j]
                code = 0
                
                # 8-neighbor LBP
                neighbors = [
                    region[i-1, j-1], region[i-1, j], region[i-1, j+1],
                    region[i, j+1], region[i+1, j+1], region[i+1, j],
                    region[i+1, j-1], region[i, j-1]
                ]
                
                for k, neighbor in enumerate(neighbors):
                    if neighbor >= center:
                        code += 2**k
                
                lbp[i, j] = code
        
        # Compute LBP histogram
        lbp_hist, _ = np.histogram(lbp, bins=256, range=(0, 256))
        
        return [
            np.mean(lbp),
            np.std(lbp),
            np.max(lbp),
            np.percentile(lbp, 50)
        ]
    
    def compute_texture_features(self, region):
        """
        Compute texture features similar to Haralick features.
        """
        # Co-occurrence matrix features (simplified)
        features = []
        
        # Compute gradients
        grad_x = cv2.Sobel(region, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(region, cv2.CV_64F, 0, 1, ksize=3)
        
        # Gradient statistics
        features.extend([
            np.mean(grad_x),
            np.mean(grad_y),
            np.std(grad_x),
            np.std(grad_y),
            np.corrcoef(grad_x.flatten(), grad_y.flatten())[0, 1] if not np.isnan(np.corrcoef(grad_x.flatten(), grad_y.flatten())[0, 1]) else 0
        ])
        
        # Edge density
        edges = cv2.Canny(region, 50, 150)
        edge_density = np.sum(edges > 0) / (region.shape[0] * region.shape[1])
        features.append(edge_density)
        
        return features
    
    def train_detection_model(self, X_train, y_train):
        """
        Train SVM model for tumor detection.
        """
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Define parameter grid for SVM
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'kernel': ['rbf', 'linear']
        }
        
        # Grid search for best parameters
        svm = SVC(random_state=42, probability=True)
        grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='f1', n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train)
        
        best_model = grid_search.best_estimator_
        
        print(f"Best SVM parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return best_model, scaler
    
    def evaluate_detection_model(self, model, scaler, X_test, y_test, title="Detection Performance"):
        """
        Evaluate tumor detection model.
        """
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        print_metrics(metrics, title)
        
        # Plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"{title} - Confusion Matrix")
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f"results/{title.lower().replace(' ', '_')}_confusion_matrix.png")
        plt.show()
        
        return metrics
    
    def count_tumors_in_image(self, model, scaler, image, annotations):
        """
        Count tumors in a single image.
        """
        predictions = []
        
        for _, row in annotations.iterrows():
            if pd.notna(row['xmin']) and pd.notna(row['ymin']) and \
               pd.notna(row['xmax']) and pd.notna(row['ymax']):
                
                xmin, ymin = int(row['xmin']), int(row['ymin'])
                xmax, ymax = int(row['xmax']), int(row['ymax'])
                
                # Extract region
                region = image[ymin:ymax, xmin:xmax]
                
                if region.size > 0:
                    # Extract features
                    features = self.extract_region_features(region)
                    
                    # Predict
                    features_scaled = scaler.transform([features])
                    prediction = model.predict(features_scaled)[0]
                    predictions.append(prediction)
        
        # Count predicted tumors
        tumor_count = sum(predictions)
        
        return tumor_count, predictions
    
    def evaluate_tumor_counting(self, model, scaler, val_files):
        """
        Evaluate tumor counting performance.
        """
        true_counts = []
        predicted_counts = []
        
        for file_info in val_files[:20]:  # Use first 20 validation files
            # Load data
            csv_path = os.path.join(self.csv_dir, file_info['csv'])
            rgb_path = os.path.join(self.rgb_dir, file_info['rgb'])
            
            image, _ = load_image_and_mask(rgb_path, None)
            annotations = load_annotations(csv_path)
            
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Count true tumors
            true_tumor_count = len(annotations[annotations['raw_classification'] == 'tumor'])
            
            # Count predicted tumors
            predicted_count, _ = self.count_tumors_in_image(model, scaler, gray, annotations)
            
            true_counts.append(true_tumor_count)
            predicted_counts.append(predicted_count)
            
            print(f"Image: {file_info['rgb']}")
            print(f"True tumors: {true_tumor_count}, Predicted tumors: {predicted_count}")
        
        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(true_counts, predicted_counts))
        
        print(f"\nTumor Counting Results:")
        print(f"RMSE: {rmse:.2f}")
        print(f"Mean absolute error: {np.mean(np.abs(np.array(true_counts) - np.array(predicted_counts))):.2f}")
        
        return {
            'rmse': rmse,
            'true_counts': true_counts,
            'predicted_counts': predicted_counts
        }
    
    def compare_masked_vs_original(self):
        """
        Compare performance using masked regions vs original image.
        """
        print("\nComparing Masked Regions vs Original Image")
        print("=" * 50)
        
        # Train model with masked regions
        print("1. Training with masked regions...")
        X_train_masked, y_train_masked, _ = self.prepare_detection_data(
            self.train_subjects[:10], use_masked_regions=True)
        X_val_masked, y_val_masked, _ = self.prepare_detection_data(
            self.val_subjects[:5], use_masked_regions=True)
        
        model_masked, scaler_masked = self.train_detection_model(X_train_masked, y_train_masked)
        metrics_masked = self.evaluate_detection_model(
            model_masked, scaler_masked, X_val_masked, y_val_masked, 
            "Masked Regions Detection")
        
        # Train model with original image
        print("\n2. Training with original image...")
        X_train_orig, y_train_orig, _ = self.prepare_detection_data(
            self.train_subjects[:10], use_masked_regions=False)
        X_val_orig, y_val_orig, _ = self.prepare_detection_data(
            self.val_subjects[:5], use_masked_regions=False)
        
        model_orig, scaler_orig = self.train_detection_model(X_train_orig, y_train_orig)
        metrics_orig = self.evaluate_detection_model(
            model_orig, scaler_orig, X_val_orig, y_val_orig, 
            "Original Image Detection")
        
        # Compare results
        print("\n3. Comparison Results:")
        print("=" * 30)
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            masked_val = metrics_masked[metric]
            orig_val = metrics_orig[metric]
            improvement = masked_val - orig_val
            print(f"{metric.capitalize()}:")
            print(f"  Masked: {masked_val:.4f}")
            print(f"  Original: {orig_val:.4f}")
            print(f"  Difference: {improvement:.4f}")
            print()
        
        return {
            'masked_metrics': metrics_masked,
            'original_metrics': metrics_orig
        }
    
    def analyze_feature_importance(self, model, feature_names=None):
        """
        Analyze which features are most important for tumor detection.
        """
        if hasattr(model, 'feature_importances_'):
            # Random Forest
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # SVM with linear kernel
            importances = np.abs(model.coef_[0])
        else:
            print("Model doesn't support feature importance analysis")
            return
        
        # Create feature names if not provided
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(importances))]
        
        # Sort features by importance
        feature_importance = list(zip(feature_names, importances))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        print("\nTop 10 Most Important Features:")
        print("=" * 40)
        for i, (feature, importance) in enumerate(feature_importance[:10]):
            print(f"{i+1}. {feature}: {importance:.4f}")
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        top_features = feature_importance[:15]
        features, importances = zip(*top_features)
        
        plt.barh(range(len(features)), importances)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Importance')
        plt.title('Top 15 Most Important Features for Tumor Detection')
        plt.tight_layout()
        plt.savefig("results/feature_importance.png")
        plt.show()
        
        return feature_importance
    
    def run_detection_pipeline(self):
        """
        Run complete tumor detection pipeline.
        """
        print("Tumor Detection Pipeline")
        print("=" * 50)
        
        # 1. Prepare training data
        print("1. Preparing training data...")
        X_train, y_train, train_counts = self.prepare_detection_data(
            self.train_subjects, use_masked_regions=True)
        
        # 2. Prepare validation data
        print("2. Preparing validation data...")
        X_val, y_val, val_counts = self.prepare_detection_data(
            self.val_subjects, use_masked_regions=True)
        
        # 3. Train detection model
        print("3. Training detection model...")
        model, scaler = self.train_detection_model(X_train, y_train)
        
        # 4. Evaluate on training data
        print("4. Evaluating on training data...")
        train_metrics = self.evaluate_detection_model(
            model, scaler, X_train, y_train, "Training Detection Performance")
        
        # 5. Evaluate on validation data
        print("5. Evaluating on validation data...")
        val_metrics = self.evaluate_detection_model(
            model, scaler, X_val, y_val, "Validation Detection Performance")
        
        # 6. Evaluate tumor counting
        print("6. Evaluating tumor counting...")
        val_files = self.get_files_for_subjects(self.val_subjects, max_files_per_subject=3)
        counting_results = self.evaluate_tumor_counting(model, scaler, val_files)
        
        # 7. Compare masked vs original
        print("7. Comparing masked vs original image...")
        comparison_results = self.compare_masked_vs_original()
        
        # 8. Analyze feature importance
        print("8. Analyzing feature importance...")
        feature_importance = self.analyze_feature_importance(model)
        
        # Save results
        all_results = {
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'counting_results': counting_results,
            'comparison_results': comparison_results,
            'feature_importance': feature_importance[:10]  # Top 10 features
        }
        
        save_results(all_results, 'detection_results.txt')
        
        print("\nDetection pipeline completed successfully!")
        print("Results saved to: results/detection_results.txt")
        
        return all_results


def main():
    """
    Main function to run tumor detection experiments.
    """
    print("Medical Image Segmentation - Tumor Detection Pipeline")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = TumorDetectionPipeline()
    
    # Run detection pipeline
    results = pipeline.run_detection_pipeline()
    
    print("\nTumor detection pipeline completed successfully!")
    print("Results saved to: results/detection_results.txt")


if __name__ == "__main__":
    main() 