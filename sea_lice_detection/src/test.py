import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.inference import load_model_for_inference, detect_sea_lice
from src.config import DATA_DIR

def test_on_synthetic_dataset():
    """
    Test the model on the synthetic dataset and calculate performance metrics
    """
    # Load model
    model = load_model_for_inference('resnet50')
    
    # Test directories
    test_fresh_dir = os.path.join(DATA_DIR, 'processed', 'test', 'fresh')
    test_infected_dir = os.path.join(DATA_DIR, 'processed', 'test', 'infected')
    
    # Get test images
    fresh_images = [os.path.join(test_fresh_dir, f) for f in os.listdir(test_fresh_dir) 
                   if f.endswith(('.jpg', '.jpeg', '.png'))]
    infected_images = [os.path.join(test_infected_dir, f) for f in os.listdir(test_infected_dir) 
                      if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Prepare for metrics calculation
    y_true = []
    y_pred = []
    y_scores = []
    
    # Process fresh images (label 0)
    print("\nTesting on fresh salmon images:")
    for img_path in fresh_images:
        try:
            prediction, _ = detect_sea_lice(img_path, model)
            y_true.append(0)
            y_pred.append(1 if prediction > 0.5 else 0)
            y_scores.append(prediction)
            print(f"Image: {os.path.basename(img_path)}, Prediction: {'Infected' if prediction > 0.5 else 'Healthy'}, Score: {prediction:.4f}")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # Process infected images (label 1)
    print("\nTesting on infected salmon images:")
    for img_path in infected_images:
        try:
            prediction, _ = detect_sea_lice(img_path, model)
            y_true.append(1)
            y_pred.append(1 if prediction > 0.5 else 0)
            y_scores.append(prediction)
            print(f"Image: {os.path.basename(img_path)}, Prediction: {'Infected' if prediction > 0.5 else 'Healthy'}, Score: {prediction:.4f}")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # Calculate metrics
    if len(y_true) > 0 and len(y_pred) > 0:
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        
        # Print metrics
        print("\nPerformance Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("Confusion Matrix:")
        print(cm)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        class_names = ['Healthy', 'Infected']
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names)
        plt.yticks(tick_marks, class_names)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig(os.path.join(DATA_DIR, '..', 'models', 'test_confusion_matrix.png'))
        
        # Compare with target metrics from proposal
        print("\nComparison with Target Metrics:")
        print(f"Target Accuracy: >95%, Achieved: {accuracy*100:.1f}%")
        print(f"Target Recall: >90%, Achieved: {recall*100:.1f}%")
        
        # Return metrics
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm
        }
    else:
        print("No valid predictions were made.")
        return None

def test_on_sample_images():
    """
    Test the model on a few sample images to demonstrate functionality
    """
    # Create a directory for sample test images if it doesn't exist
    sample_dir = os.path.join(DATA_DIR, 'samples')
    os.makedirs(sample_dir, exist_ok=True)
    
    # Create a few more synthetic test images
    create_additional_test_images(sample_dir)
    
    # Load model
    model = load_model_for_inference('resnet50')
    
    # Get sample images
    sample_images = [os.path.join(sample_dir, f) for f in os.listdir(sample_dir) 
                    if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Process each image
    print("\nTesting on sample images:")
    for img_path in sample_images:
        try:
            prediction, image = detect_sea_lice(img_path, model)
            result = 'Infected' if prediction > 0.5 else 'Healthy'
            confidence = prediction if prediction > 0.5 else 1 - prediction
            print(f"Image: {os.path.basename(img_path)}, Prediction: {result}, Confidence: {confidence*100:.1f}%")
            
            # Save visualization
            save_path = os.path.join(sample_dir, f"{os.path.splitext(os.path.basename(img_path))[0]}_result.png")
            
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
            plt.savefig(save_path)
            plt.close()
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

def create_additional_test_images(output_dir):
    """
    Create a few more synthetic test images for demonstration
    """
    import cv2
    
    # Create a healthy salmon image
    img_healthy = np.ones((250, 600, 3), dtype=np.uint8)
    img_healthy[:, :, 0] = np.random.randint(30, 60)  # B channel
    img_healthy[:, :, 1] = np.random.randint(100, 150)  # G channel
    img_healthy[:, :, 2] = np.random.randint(180, 230)  # R channel
    
    # Add fish-like shape
    pts = np.array([
        [100, 125], 
        [500, 75], 
        [500, 175], 
        [100, 125]
    ], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(img_healthy, [pts], (40, 120, 200))
    
    # Add eye
    cv2.circle(img_healthy, (150, 125), 10, (10, 10, 10), -1)
    
    # Save the image
    cv2.imwrite(os.path.join(output_dir, "sample_healthy.jpg"), img_healthy)
    
    # Create an infected salmon image
    img_infected = np.ones((250, 600, 3), dtype=np.uint8)
    img_infected[:, :, 0] = np.random.randint(30, 60)  # B channel
    img_infected[:, :, 1] = np.random.randint(100, 150)  # G channel
    img_infected[:, :, 2] = np.random.randint(180, 230)  # R channel
    
    # Add fish-like shape
    pts = np.array([
        [100, 125], 
        [500, 75], 
        [500, 175], 
        [100, 125]
    ], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(img_infected, [pts], (40, 120, 200))
    
    # Add eye
    cv2.circle(img_infected, (150, 125), 10, (10, 10, 10), -1)
    
    # Add sea lice (white/gray spots)
    for _ in range(10):
        x = np.random.randint(150, 500)
        y = np.random.randint(75, 175)
        radius = np.random.randint(3, 8)
        color = (
            np.random.randint(200, 255),
            np.random.randint(200, 255),
            np.random.randint(200, 255)
        )
        cv2.circle(img_infected, (x, y), radius, color, -1)
    
    # Save the image
    cv2.imwrite(os.path.join(output_dir, "sample_infected.jpg"), img_infected)
    
    print(f"Created additional test images in {output_dir}")

if __name__ == "__main__":
    # Test on synthetic dataset
    print("Testing on synthetic dataset...")
    metrics = test_on_synthetic_dataset()
    
    # Test on sample images
    print("\nTesting on sample images...")
    test_on_sample_images()
    
    print("\nTesting completed successfully!")
