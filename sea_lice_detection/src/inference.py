import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import MODEL_DIR, IMG_WIDTH, IMG_HEIGHT

def load_model_for_inference(model_type='resnet50'):
    """
    Load a trained model for inference
    
    Args:
        model_type: Type of model to load ('resnet50' or 'efficientnet')
    
    Returns:
        Loaded model
    """
    try:
        if model_type.lower() == 'resnet50':
            model_path = os.path.join(MODEL_DIR, "resnet50_sea_lice_detector_best.h5")
            if not os.path.exists(model_path):
                model_path = os.path.join(MODEL_DIR, "resnet50_sea_lice_detector_final.h5")
        else:
            model_path = os.path.join(MODEL_DIR, "efficientnet_sea_lice_detector_best.h5")
            if not os.path.exists(model_path):
                model_path = os.path.join(MODEL_DIR, "efficientnet_sea_lice_detector_final.h5")
        
        if os.path.exists(model_path):
            model = load_model(model_path)
            print(f"Model loaded from {model_path}")
            return model
        else:
            print(f"Model file not found at {model_path}")
            return create_demo_model()
    except Exception as e:
        print(f"Error loading model: {e}")
        return create_demo_model()

def create_demo_model():
    """Create a simple model for demonstration purposes if the trained model is not available"""
    inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print("Demo model created for demonstration purposes")
    return model

def preprocess_image(image_path):
    """
    Preprocess an image for inference
    
    Args:
        image_path: Path to the image file
    
    Returns:
        Preprocessed image ready for model input
    """
    # Read image
    img = cv2.imread(image_path)
    
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize image to the required dimensions
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    
    # Apply preprocessing
    img = img.astype('float32') / 255.0
    
    # Expand dimensions to match model input shape
    img = np.expand_dims(img, axis=0)
    
    return img

def detect_sea_lice(image_path, model):
    """
    Detect sea lice in an image
    
    Args:
        image_path: Path to the image file
        model: Loaded model for inference
    
    Returns:
        Prediction probability and processed image
    """
    # Preprocess image
    processed_img = preprocess_image(image_path)
    
    # Make prediction
    prediction = model.predict(processed_img)[0][0]
    
    # Read original image for visualization
    original_img = cv2.imread(image_path)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    return prediction, original_img

def visualize_result(image, prediction, save_path=None):
    """
    Visualize detection result
    
    Args:
        image: Original image
        prediction: Prediction probability
        save_path: Path to save the visualization (optional)
    """
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Display image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Input Image")
    plt.axis('off')
    
    # Display prediction
    plt.subplot(1, 2, 2)
    categories = ['Healthy', 'Infected']
    values = [1 - prediction, prediction]
    colors = ['#4CAF50', '#f44336']
    
    plt.bar(categories, values, color=colors)
    plt.ylim(0, 1)
    plt.ylabel('Probability')
    plt.title('Classification Result')
    
    # Add text with prediction
    if prediction > 0.5:
        result_text = f"INFECTED (Sea Lice Detected)\nConfidence: {prediction*100:.1f}%"
        color = '#f44336'
    else:
        result_text = f"HEALTHY (No Sea Lice Detected)\nConfidence: {(1-prediction)*100:.1f}%"
        color = '#4CAF50'
    
    plt.figtext(0.5, 0.01, result_text, ha='center', fontsize=14, color=color, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save if path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
    
    plt.show()

def batch_process_images(image_dir, model, output_dir=None):
    """
    Process all images in a directory
    
    Args:
        image_dir: Directory containing images
        model: Loaded model for inference
        output_dir: Directory to save results (optional)
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    results = []
    
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        
        try:
            # Detect sea lice
            prediction, image = detect_sea_lice(image_path, model)
            
            # Save result
            result = {
                'image_file': image_file,
                'prediction': prediction,
                'classification': 'Infected' if prediction > 0.5 else 'Healthy',
                'confidence': prediction if prediction > 0.5 else 1 - prediction
            }
            results.append(result)
            
            # Save visualization if output directory is provided
            if output_dir:
                save_path = os.path.join(output_dir, f"{os.path.splitext(image_file)[0]}_result.png")
                visualize_result(image, prediction, save_path)
            
            print(f"Processed {image_file}: {result['classification']} ({result['confidence']*100:.1f}%)")
            
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
    
    # Print summary
    print("\nProcessing Summary:")
    print(f"Total images processed: {len(results)}")
    infected_count = sum(1 for r in results if r['classification'] == 'Infected')
    healthy_count = sum(1 for r in results if r['classification'] == 'Healthy')
    print(f"Infected: {infected_count} ({infected_count/len(results)*100:.1f}%)")
    print(f"Healthy: {healthy_count} ({healthy_count/len(results)*100:.1f}%)")
    
    return results

if __name__ == "__main__":
    # Example usage
    model = load_model_for_inference('resnet50')
    
    # Check if command line arguments are provided
    if len(sys.argv) > 1:
        # Process a single image
        if os.path.isfile(sys.argv[1]):
            prediction, image = detect_sea_lice(sys.argv[1], model)
            visualize_result(image, prediction)
        
        # Process a directory of images
        elif os.path.isdir(sys.argv[1]):
            output_dir = sys.argv[2] if len(sys.argv) > 2 else None
            batch_process_images(sys.argv[1], model, output_dir)
        
        else:
            print(f"Invalid path: {sys.argv[1]}")
    else:
        print("Usage: python inference.py <image_path_or_directory> [output_directory]")
