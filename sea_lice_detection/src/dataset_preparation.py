import os
import sys
import requests
import zipfile
import shutil
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATA_DIR

def download_sample_dataset():
    """
    Since we can't directly download the SalmonScan dataset from Mendeley,
    we'll create a small sample dataset for demonstration purposes.
    
    This function will download some sample salmon images from the internet
    to simulate the SalmonScan dataset structure.
    """
    # Create raw data directories
    os.makedirs(os.path.join(DATA_DIR, 'raw', 'fresh'), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, 'raw', 'infected'), exist_ok=True)
    
    # Sample image URLs (replace with actual salmon images if available)
    fresh_salmon_urls = [
        "https://www.seafoodsource.com/images/news/salmon_fresh_AdobeStock_46209896.jpeg",
        "https://cdn.britannica.com/79/65979-050-5C6DA36A/Sockeye-salmon.jpg",
        "https://www.thespruceeats.com/thmb/HgM2h-VfpZG-cUgQlQRSLRm_KcQ=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/atlantic-salmon-fish-1300561-hero-01-cbfb489d27b94ca0a5af8d8a6c62f9b1.jpg"
    ]
    
    infected_salmon_urls = [
        "https://www.fishfarmingexpert.com/wp-content/uploads/sites/3/2022/03/Sea-lice-on-salmon-Marine-Scotland.jpg",
        "https://www.intrafish.com/incoming/2023/03/15/sea-lice-on-salmon-marine-scotland-science/2-1-1398314.jpg",
        "https://www.fishfarmingexpert.com/wp-content/uploads/sites/3/2022/01/Sea-lice-on-salmon-Marine-Scotland-Science.jpg"
    ]
    
    # Download fresh salmon images
    for i, url in enumerate(fresh_salmon_urls):
        try:
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(os.path.join(DATA_DIR, 'raw', 'fresh', f'fresh_{i+1}.jpg'), 'wb') as f:
                    f.write(response.content)
                print(f"Downloaded fresh salmon image {i+1}")
            else:
                print(f"Failed to download image from {url}")
        except Exception as e:
            print(f"Error downloading image: {e}")
    
    # Download infected salmon images
    for i, url in enumerate(infected_salmon_urls):
        try:
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(os.path.join(DATA_DIR, 'raw', 'infected', f'infected_{i+1}.jpg'), 'wb') as f:
                    f.write(response.content)
                print(f"Downloaded infected salmon image {i+1}")
            else:
                print(f"Failed to download image from {url}")
        except Exception as e:
            print(f"Error downloading image: {e}")
    
    print("Sample dataset download complete!")

def preprocess_images():
    """
    Preprocess the raw images and organize them into train, validation, and test sets
    """
    # Create processed data directories
    os.makedirs(os.path.join(DATA_DIR, 'processed', 'train', 'fresh'), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, 'processed', 'train', 'infected'), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, 'processed', 'val', 'fresh'), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, 'processed', 'val', 'infected'), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, 'processed', 'test', 'fresh'), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, 'processed', 'test', 'infected'), exist_ok=True)
    
    # Process fresh salmon images
    fresh_images = [f for f in os.listdir(os.path.join(DATA_DIR, 'raw', 'fresh')) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Split into train, validation, and test sets (70%, 15%, 15%)
    fresh_train, fresh_temp = train_test_split(fresh_images, test_size=0.3, random_state=42)
    fresh_val, fresh_test = train_test_split(fresh_temp, test_size=0.5, random_state=42)
    
    # Process infected salmon images
    infected_images = [f for f in os.listdir(os.path.join(DATA_DIR, 'raw', 'infected')) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Split into train, validation, and test sets (70%, 15%, 15%)
    infected_train, infected_temp = train_test_split(infected_images, test_size=0.3, random_state=42)
    infected_val, infected_test = train_test_split(infected_temp, test_size=0.5, random_state=42)
    
    # Process and save fresh salmon images
    for img_set, dest_folder in [
        (fresh_train, os.path.join(DATA_DIR, 'processed', 'train', 'fresh')),
        (fresh_val, os.path.join(DATA_DIR, 'processed', 'val', 'fresh')),
        (fresh_test, os.path.join(DATA_DIR, 'processed', 'test', 'fresh'))
    ]:
        for img_name in img_set:
            img_path = os.path.join(DATA_DIR, 'raw', 'fresh', img_name)
            process_and_save_image(img_path, os.path.join(dest_folder, img_name))
    
    # Process and save infected salmon images
    for img_set, dest_folder in [
        (infected_train, os.path.join(DATA_DIR, 'processed', 'train', 'infected')),
        (infected_val, os.path.join(DATA_DIR, 'processed', 'val', 'infected')),
        (infected_test, os.path.join(DATA_DIR, 'processed', 'test', 'infected'))
    ]:
        for img_name in img_set:
            img_path = os.path.join(DATA_DIR, 'raw', 'infected', img_name)
            process_and_save_image(img_path, os.path.join(dest_folder, img_name))
    
    print("Image preprocessing complete!")

def process_and_save_image(src_path, dest_path):
    """
    Process a single image and save it to the destination path
    """
    try:
        # Read image
        img = cv2.imread(src_path)
        
        if img is None:
            print(f"Failed to read image: {src_path}")
            return
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize image to 600x250 as specified in the dataset description
        img = cv2.resize(img, (600, 250))
        
        # Apply noise reduction
        img = cv2.GaussianBlur(img, (3, 3), 0)
        
        # Apply contrast enhancement
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        
        # Convert back to BGR for saving with OpenCV
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Save processed image
        cv2.imwrite(dest_path, img)
    except Exception as e:
        print(f"Error processing image {src_path}: {e}")

def augment_training_data():
    """
    Apply data augmentation to the training set to increase its size
    """
    train_fresh_dir = os.path.join(DATA_DIR, 'processed', 'train', 'fresh')
    train_infected_dir = os.path.join(DATA_DIR, 'processed', 'train', 'infected')
    
    # Augment fresh salmon images
    fresh_images = [f for f in os.listdir(train_fresh_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    for img_name in fresh_images:
        img_path = os.path.join(train_fresh_dir, img_name)
        img = cv2.imread(img_path)
        
        if img is None:
            continue
        
        # Horizontal flip
        h_flip = cv2.flip(img, 1)
        cv2.imwrite(os.path.join(train_fresh_dir, f'h_flip_{img_name}'), h_flip)
        
        # Vertical flip
        v_flip = cv2.flip(img, 0)
        cv2.imwrite(os.path.join(train_fresh_dir, f'v_flip_{img_name}'), v_flip)
        
        # Rotation (90 degrees)
        rows, cols = img.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), 90, 1)
        rotated = cv2.warpAffine(img, rotation_matrix, (cols, rows))
        cv2.imwrite(os.path.join(train_fresh_dir, f'rot90_{img_name}'), rotated)
    
    # Augment infected salmon images
    infected_images = [f for f in os.listdir(train_infected_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    for img_name in infected_images:
        img_path = os.path.join(train_infected_dir, img_name)
        img = cv2.imread(img_path)
        
        if img is None:
            continue
        
        # Horizontal flip
        h_flip = cv2.flip(img, 1)
        cv2.imwrite(os.path.join(train_infected_dir, f'h_flip_{img_name}'), h_flip)
        
        # Vertical flip
        v_flip = cv2.flip(img, 0)
        cv2.imwrite(os.path.join(train_infected_dir, f'v_flip_{img_name}'), v_flip)
        
        # Rotation (90 degrees)
        rows, cols = img.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), 90, 1)
        rotated = cv2.warpAffine(img, rotation_matrix, (cols, rows))
        cv2.imwrite(os.path.join(train_infected_dir, f'rot90_{img_name}'), rotated)
    
    print("Data augmentation complete!")

if __name__ == "__main__":
    download_sample_dataset()
    preprocess_images()
    augment_training_data()
    print("Dataset preprocessing completed successfully!")
