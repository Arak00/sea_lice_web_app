import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import matplotlib.pyplot as plt

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import IMG_WIDTH, IMG_HEIGHT, DATA_DIR

def create_dataset_structure():
    """
    Create the necessary directory structure for the dataset
    """
    os.makedirs(os.path.join(DATA_DIR, 'raw', 'fresh'), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, 'raw', 'infected'), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, 'processed', 'train', 'fresh'), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, 'processed', 'train', 'infected'), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, 'processed', 'val', 'fresh'), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, 'processed', 'val', 'infected'), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, 'processed', 'test', 'fresh'), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, 'processed', 'test', 'infected'), exist_ok=True)
    
    print("Dataset directory structure created successfully!")

def preprocess_image(image_path):
    """
    Preprocess a single image
    """
    # Read image
    img = cv2.imread(image_path)
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize image to the required dimensions
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    
    # Normalize pixel values to [0, 1]
    img = img / 255.0
    
    return img

def data_augmentation():
    """
    Create data augmentation pipeline
    """
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )
    
    # Validation data should not be augmented
    val_datagen = ImageDataGenerator()
    
    return train_datagen, val_datagen

def load_and_preprocess_dataset():
    """
    Load and preprocess the dataset
    """
    train_datagen, val_datagen = data_augmentation()
    
    train_generator = train_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'processed', 'train'),
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=32,
        class_mode='binary'
    )
    
    val_generator = val_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'processed', 'val'),
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=32,
        class_mode='binary'
    )
    
    test_generator = val_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'processed', 'test'),
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=32,
        class_mode='binary'
    )
    
    return train_generator, val_generator, test_generator

if __name__ == "__main__":
    create_dataset_structure()
    print("Data preprocessing module is ready!")
