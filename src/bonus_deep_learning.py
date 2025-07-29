"""
Bonus Deep Learning Module for Medical Image Segmentation Project
Section 6: Deep Learning Segmentation on TCGA-EW-A1P1-DX1
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import seaborn as sns
from utils import (
    load_image_and_mask, load_annotations, calculate_metrics,
    print_metrics, save_results, calculate_dice_coefficient
)


class DeepLearningSegmentation:
    def __init__(self, data_dir="Data"):
        self.data_dir = data_dir
        self.csv_dir = os.path.join(data_dir, "csv")
        self.rgb_dir = os.path.join(data_dir, "rgb")
        self.mask_dir = os.path.join(data_dir, "mask")
        
        # Target subject for bonus question
        self.target_subject = "TCGA-EW-A1P1-DX1"
        
        # Create output directories
        os.makedirs("results", exist_ok=True)
        
        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)
        
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
    
    def prepare_deep_learning_data(self, subject_id):
        """
        Prepare data for deep learning segmentation.
        """
        files = self.get_files_for_subject(subject_id)
        
        if len(files) == 0:
            print(f"No files found for subject {subject_id}")
            return None, None, None, None
        
        print(f"Found {len(files)} files for subject {subject_id}")
        
        images = []
        masks = []
        
        for file_info in files:
            try:
                # Load image and mask
                rgb_path = os.path.join(self.rgb_dir, file_info['rgb'])
                mask_path = os.path.join(self.mask_dir, file_info['mask'])
                
                image, mask = load_image_and_mask(rgb_path, mask_path)
                
                # Resize to standard size for deep learning
                target_size = (256, 256)
                image_resized = cv2.resize(image, target_size)
                mask_resized = cv2.resize(mask, target_size)
                
                # Normalize image
                image_normalized = image_resized.astype(np.float32) / 255.0
                
                # Create binary mask (nuclei vs background)
                mask_binary = (mask_resized > 0).astype(np.float32)
                
                images.append(image_normalized)
                masks.append(mask_binary)
                
            except Exception as e:
                print(f"Error processing {file_info['rgb']}: {e}")
                continue
        
        if len(images) == 0:
            print("No valid images found")
            return None, None, None, None
        
        # Convert to numpy arrays
        X = np.array(images)
        y = np.array(masks)
        
        print(f"Prepared {len(X)} images with shape {X.shape}")
        print(f"Prepared {len(y)} masks with shape {y.shape}")
        
        # Split into train (30 samples) and test (8 samples)
        if len(X) >= 38:
            # Use exactly 30 for training, 8 for testing
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=8, random_state=42, stratify=None
            )
        else:
            # Use 80% train, 20% test if we have fewer than 38 samples
            train_size = int(0.8 * len(X))
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test
    
    def create_unet_model(self, input_shape=(256, 256, 3)):
        """
        Create a U-Net model for image segmentation.
        """
        # Encoder (downsampling path)
        inputs = layers.Input(input_shape)
        
        # Encoder blocks
        conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
        conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
        pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
        
        conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
        conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
        pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
        
        conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
        conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
        pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
        
        # Bridge
        conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(pool3)
        conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv4)
        
        # Decoder (upsampling path)
        up5 = layers.UpSampling2D(size=(2, 2))(conv4)
        up5 = layers.concatenate([up5, conv3])
        conv5 = layers.Conv2D(256, 3, activation='relu', padding='same')(up5)
        conv5 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv5)
        
        up6 = layers.UpSampling2D(size=(2, 2))(conv5)
        up6 = layers.concatenate([up6, conv2])
        conv6 = layers.Conv2D(128, 3, activation='relu', padding='same')(up6)
        conv6 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv6)
        
        up7 = layers.UpSampling2D(size=(2, 2))(conv6)
        up7 = layers.concatenate([up7, conv1])
        conv7 = layers.Conv2D(64, 3, activation='relu', padding='same')(up7)
        conv7 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv7)
        
        # Output layer
        outputs = layers.Conv2D(1, 1, activation='sigmoid')(conv7)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        return model
    
    def create_transfer_learning_model(self, input_shape=(256, 256, 3)):
        """
        Create a model using transfer learning with pre-trained VGG16.
        """
        # Load pre-trained VGG16 (without top layers)
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
        
        # Freeze the base model
        base_model.trainable = False
        
        # Create the model
        inputs = layers.Input(input_shape)
        
        # Preprocess input for VGG16
        x = layers.Lambda(lambda x: tf.keras.applications.vgg16.preprocess_input(x))(inputs)
        
        # Use VGG16 features
        x = base_model(x, training=False)
        
        # Add custom layers for segmentation
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Reshape for segmentation output
        x = layers.Dense(256 * 256, activation='relu')(x)
        outputs = layers.Reshape((256, 256, 1))(x)
        outputs = layers.Activation('sigmoid')(outputs)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        return model
    
    def create_data_generator(self, X, y, batch_size=4):
        """
        Create data generator with augmentation for training.
        """
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=0.1,
            fill_mode='nearest'
        )
        
        return datagen.flow(X, y, batch_size=batch_size)
    
    def dice_coefficient_loss(self, y_true, y_pred):
        """
        Custom loss function using Dice coefficient.
        """
        smooth = 1e-6
        y_true_f = tf.keras.backend.flatten(y_true)
        y_pred_f = tf.keras.backend.flatten(y_pred)
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        return 1 - (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
    
    def dice_coefficient_metric(self, y_true, y_pred):
        """
        Dice coefficient as a metric.
        """
        smooth = 1e-6
        y_true_f = tf.keras.backend.flatten(y_true)
        y_pred_f = tf.keras.backend.flatten(y_pred)
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
    
    def train_model(self, X_train, y_train, X_test, y_test, model_type='unet'):
        """
        Train the deep learning model.
        """
        print(f"\nTraining {model_type.upper()} model...")
        
        # Create model
        if model_type == 'unet':
            model = self.create_unet_model()
        elif model_type == 'transfer':
            model = self.create_transfer_learning_model()
        else:
            raise ValueError("model_type must be 'unet' or 'transfer'")
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=self.dice_coefficient_loss,
            metrics=['accuracy', self.dice_coefficient_metric]
        )
        
        print(f"Model summary:")
        model.summary()
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7)
        ]
        
        # Data generator for training
        train_generator = self.create_data_generator(X_train, y_train, batch_size=4)
        
        # Train model
        history = model.fit(
            train_generator,
            steps_per_epoch=len(X_train) // 4,
            epochs=50,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        return model, history
    
    def evaluate_model(self, model, X_test, y_test, title="Deep Learning Segmentation"):
        """
        Evaluate the trained model.
        """
        # Predict on test set
        y_pred = model.predict(X_test)
        
        # Convert to binary predictions
        y_pred_binary = (y_pred > 0.5).astype(np.float32)
        
        # Calculate metrics for each image
        dice_scores = []
        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        
        for i in range(len(X_test)):
            # Flatten for metric calculation
            y_true_flat = y_test[i].flatten()
            y_pred_flat = y_pred_binary[i].flatten()
            
            # Calculate metrics
            dice = calculate_dice_coefficient(y_true_flat, y_pred_flat)
            accuracy = accuracy_score(y_true_flat, y_pred_flat)
            precision = precision_score(y_true_flat, y_pred_flat, zero_division=0)
            recall = recall_score(y_true_flat, y_pred_flat, zero_division=0)
            f1 = f1_score(y_true_flat, y_pred_flat, zero_division=0)
            
            dice_scores.append(dice)
            accuracy_scores.append(accuracy)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
        
        # Average metrics
        avg_metrics = {
            'dice_coefficient': np.mean(dice_scores),
            'accuracy': np.mean(accuracy_scores),
            'precision': np.mean(precision_scores),
            'recall': np.mean(recall_scores),
            'f1_score': np.mean(f1_scores)
        }
        
        print_metrics(avg_metrics, title)
        
        # Plot training history
        if hasattr(self, 'history'):
            self.plot_training_history()
        
        # Visualize results
        self.visualize_results(X_test, y_test, y_pred_binary, title)
        
        return avg_metrics, y_pred_binary
    
    def plot_training_history(self):
        """
        Plot training history.
        """
        if not hasattr(self, 'history'):
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        # Accuracy
        axes[0, 1].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0, 1].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        
        # Dice Coefficient
        axes[1, 0].plot(self.history.history['dice_coefficient_metric'], label='Training Dice')
        axes[1, 0].plot(self.history.history['val_dice_coefficient_metric'], label='Validation Dice')
        axes[1, 0].set_title('Dice Coefficient')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Dice Coefficient')
        axes[1, 0].legend()
        
        # Learning Rate
        if 'lr' in self.history.history:
            axes[1, 1].plot(self.history.history['lr'], label='Learning Rate')
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig("results/deep_learning_training_history.png")
        plt.show()
    
    def visualize_results(self, X_test, y_test, y_pred, title):
        """
        Visualize segmentation results.
        """
        n_samples = min(6, len(X_test))
        fig, axes = plt.subplots(n_samples, 3, figsize=(15, 5*n_samples))
        
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_samples):
            # Original image
            axes[i, 0].imshow(X_test[i])
            axes[i, 0].set_title(f'Original Image {i+1}')
            axes[i, 0].axis('off')
            
            # Ground truth mask
            axes[i, 1].imshow(y_test[i].squeeze(), cmap='gray')
            axes[i, 1].set_title(f'Ground Truth {i+1}')
            axes[i, 1].axis('off')
            
            # Predicted mask
            axes[i, 2].imshow(y_pred[i].squeeze(), cmap='gray')
            axes[i, 2].set_title(f'Predicted Mask {i+1}')
            axes[i, 2].axis('off')
        
        plt.suptitle(f'{title} - Sample Results')
        plt.tight_layout()
        plt.savefig(f"results/{title.lower().replace(' ', '_')}_results.png")
        plt.show()
    
    def run_deep_learning_pipeline(self):
        """
        Run complete deep learning segmentation pipeline.
        """
        print("Deep Learning Segmentation Pipeline")
        print("=" * 50)
        print(f"Target Subject: {self.target_subject}")
        
        # 1. Prepare data
        print("\n1. Preparing data...")
        data = self.prepare_deep_learning_data(self.target_subject)
        
        if data[0] is None:
            print("Failed to prepare data. Exiting.")
            return None
        
        X_train, X_test, y_train, y_test = data
        
        # 2. Train U-Net model
        print("\n2. Training U-Net model...")
        unet_model, unet_history = self.train_model(X_train, y_train, X_test, y_test, 'unet')
        self.history = unet_history
        
        # 3. Evaluate U-Net
        print("\n3. Evaluating U-Net model...")
        unet_metrics, unet_predictions = self.evaluate_model(
            unet_model, X_test, y_test, "U-Net Segmentation")
        
        # 4. Train Transfer Learning model
        print("\n4. Training Transfer Learning model...")
        transfer_model, transfer_history = self.train_model(X_train, y_train, X_test, y_test, 'transfer')
        
        # 5. Evaluate Transfer Learning
        print("\n5. Evaluating Transfer Learning model...")
        transfer_metrics, transfer_predictions = self.evaluate_model(
            transfer_model, X_test, y_test, "Transfer Learning Segmentation")
        
        # 6. Compare models
        print("\n6. Model Comparison:")
        print("=" * 30)
        for metric in ['dice_coefficient', 'accuracy', 'precision', 'recall', 'f1_score']:
            unet_val = unet_metrics[metric]
            transfer_val = transfer_metrics[metric]
            print(f"{metric.capitalize()}:")
            print(f"  U-Net: {unet_val:.4f}")
            print(f"  Transfer Learning: {transfer_val:.4f}")
            print()
        
        # Save results
        results = {
            'target_subject': self.target_subject,
            'unet_metrics': unet_metrics,
            'transfer_metrics': transfer_metrics,
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        save_results(results, 'deep_learning_results.txt')
        
        print("\nDeep learning pipeline completed successfully!")
        print("Results saved to: results/deep_learning_results.txt")
        
        return results


def main():
    """
    Main function to run deep learning segmentation experiments.
    """
    print("Medical Image Segmentation - Deep Learning Bonus")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = DeepLearningSegmentation()
    
    # Run deep learning pipeline
    results = pipeline.run_deep_learning_pipeline()
    
    if results:
        print("\nDeep learning segmentation completed successfully!")
        print("Results saved to: results/deep_learning_results.txt")
    else:
        print("\nDeep learning segmentation failed.")


if __name__ == "__main__":
    main() 