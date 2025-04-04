import os
import sys
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATA_DIR, IMG_WIDTH, IMG_HEIGHT

def create_synthetic_dataset():
    """
    Create a synthetic dataset of salmon images for demonstration purposes.
    Since we couldn't download real images, we'll generate synthetic ones.
    """
    # Create raw data directories
    os.makedirs(os.path.join(DATA_DIR, 'raw', 'fresh'), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, 'raw', 'infected'), exist_ok=True)
    
    # Generate synthetic images
    # Fresh salmon: mostly orange/pink color
    for i in range(10):
        # Create a base orange/pink image for fresh salmon
        img = np.ones((250, 600, 3), dtype=np.uint8)
        img[:, :, 0] = np.random.randint(30, 60)  # B channel
        img[:, :, 1] = np.random.randint(100, 150)  # G channel
        img[:, :, 2] = np.random.randint(180, 230)  # R channel
        
        # Add some texture/pattern
        for _ in range(50):
            x = np.random.randint(0, 600)
            y = np.random.randint(0, 250)
            radius = np.random.randint(5, 20)
            color = (
                np.random.randint(20, 50),
                np.random.randint(90, 140),
                np.random.randint(170, 220)
            )
            cv2.circle(img, (x, y), radius, color, -1)
        
        # Add fish-like shape
        pts = np.array([
            [100, 125], 
            [500, 75], 
            [500, 175], 
            [100, 125]
        ], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(img, [pts], (40, 120, 200))
        
        # Add eye
        cv2.circle(img, (150, 125), 10, (10, 10, 10), -1)
        
        # Save the image
        cv2.imwrite(os.path.join(DATA_DIR, 'raw', 'fresh', f'fresh_{i+1}.jpg'), img)
        print(f"Created synthetic fresh salmon image {i+1}")
    
    # Infected salmon: similar to fresh but with white/gray spots (sea lice)
    for i in range(15):
        # Create a base orange/pink image for infected salmon
        img = np.ones((250, 600, 3), dtype=np.uint8)
        img[:, :, 0] = np.random.randint(30, 60)  # B channel
        img[:, :, 1] = np.random.randint(100, 150)  # G channel
        img[:, :, 2] = np.random.randint(180, 230)  # R channel
        
        # Add some texture/pattern
        for _ in range(50):
            x = np.random.randint(0, 600)
            y = np.random.randint(0, 250)
            radius = np.random.randint(5, 20)
            color = (
                np.random.randint(20, 50),
                np.random.randint(90, 140),
                np.random.randint(170, 220)
            )
            cv2.circle(img, (x, y), radius, color, -1)
        
        # Add fish-like shape
        pts = np.array([
            [100, 125], 
            [500, 75], 
            [500, 175], 
            [100, 125]
        ], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(img, [pts], (40, 120, 200))
        
        # Add eye
        cv2.circle(img, (150, 125), 10, (10, 10, 10), -1)
        
        # Add sea lice (white/gray spots)
        for _ in range(np.random.randint(5, 15)):
            x = np.random.randint(150, 500)
            y = np.random.randint(75, 175)
            radius = np.random.randint(3, 8)
            color = (
                np.random.randint(200, 255),
                np.random.randint(200, 255),
                np.random.randint(200, 255)
            )
            cv2.circle(img, (x, y), radius, color, -1)
        
        # Save the image
        cv2.imwrite(os.path.join(DATA_DIR, 'raw', 'infected', f'infected_{i+1}.jpg'), img)
        print(f"Created synthetic infected salmon image {i+1}")
    
    print("Synthetic dataset creation complete!")

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
        
        # Apply noise reduction
        img = cv2.GaussianBlur(img, (3, 3), 0)
        
        # Apply contrast enhancement
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
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
    create_synthetic_dataset()
    preprocess_images()
    augment_training_data()
    print("Dataset preprocessing completed successfully!")
