import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50, EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATA_DIR, MODEL_DIR, IMG_WIDTH, IMG_HEIGHT, BATCH_SIZE, EPOCHS

def create_data_generators():
    """
    Create data generators for training, validation, and testing
    """
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )
    
    # Only rescaling for validation and test data
    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'processed', 'train'),
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'processed', 'val'),
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )
    
    test_generator = test_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'processed', 'test'),
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )
    
    return train_generator, val_generator, test_generator

def build_custom_cnn_model():
    """
    Build a custom CNN model for sea lice detection
    """
    model = models.Sequential([
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # Fourth convolutional block
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

def build_transfer_learning_model(base_model_name='resnet50'):
    """
    Build a transfer learning model using pre-trained weights
    
    Args:
        base_model_name: Name of the base model to use ('resnet50' or 'efficientnet')
    
    Returns:
        A compiled Keras model
    """
    if base_model_name.lower() == 'resnet50':
        # Load ResNet50 with pre-trained ImageNet weights
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
        )
    elif base_model_name.lower() == 'efficientnet':
        # Load EfficientNetB0 with pre-trained ImageNet weights
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
        )
    else:
        raise ValueError("base_model_name must be 'resnet50' or 'efficientnet'")
    
    # Freeze the base model layers
    base_model.trainable = False
    
    # Create the model
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

def train_model(model, train_generator, val_generator, model_name='sea_lice_detector'):
    """
    Train the model and save it
    
    Args:
        model: The compiled Keras model
        train_generator: Training data generator
        val_generator: Validation data generator
        model_name: Name to use when saving the model
    
    Returns:
        Training history
    """
    # Create callbacks
    checkpoint_path = os.path.join(MODEL_DIR, f"{model_name}_best.h5")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    # Train the model
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=[checkpoint_callback, early_stopping]
    )
    
    # Save the final model
    model.save(os.path.join(MODEL_DIR, f"{model_name}_final.h5"))
    
    return history

def evaluate_model(model, test_generator):
    """
    Evaluate the model on the test set
    
    Args:
        model: The trained Keras model
        test_generator: Test data generator
    
    Returns:
        Dictionary of evaluation metrics
    """
    # Get predictions
    test_generator.reset()
    y_pred_prob = model.predict(test_generator)
    y_pred = (y_pred_prob > 0.5).astype(int)
    y_true = test_generator.classes
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    # Print results
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    class_names = ['Fresh', 'Infected']
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
    plt.savefig(os.path.join(MODEL_DIR, 'confusion_matrix.png'))
    
    # Plot training history if available
    if hasattr(model, 'history') and model.history is not None:
        history = model.history.history
        plt.figure(figsize=(12, 4))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history['accuracy'])
        plt.plot(history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_DIR, 'training_history.png'))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm
    }

if __name__ == "__main__":
    # Create data generators
    train_generator, val_generator, test_generator = create_data_generators()
    
    # Build and train the transfer learning model with ResNet50
    print("Building and training ResNet50 transfer learning model...")
    resnet_model = build_transfer_learning_model('resnet50')
    resnet_history = train_model(resnet_model, train_generator, val_generator, 'resnet50_sea_lice_detector')
    
    # Evaluate the ResNet50 model
    print("\nEvaluating ResNet50 model...")
    resnet_metrics = evaluate_model(resnet_model, test_generator)
    
    # Build and train the transfer learning model with EfficientNet
    print("\nBuilding and training EfficientNet transfer learning model...")
    efficientnet_model = build_transfer_learning_model('efficientnet')
    efficientnet_history = train_model(efficientnet_model, train_generator, val_generator, 'efficientnet_sea_lice_detector')
    
    # Evaluate the EfficientNet model
    print("\nEvaluating EfficientNet model...")
    efficientnet_metrics = evaluate_model(efficientnet_model, test_generator)
    
    # Compare models
    print("\nModel Comparison:")
    print(f"ResNet50 Accuracy: {resnet_metrics['accuracy']:.4f}, Precision: {resnet_metrics['precision']:.4f}, Recall: {resnet_metrics['recall']:.4f}, F1: {resnet_metrics['f1_score']:.4f}")
    print(f"EfficientNet Accuracy: {efficientnet_metrics['accuracy']:.4f}, Precision: {efficientnet_metrics['precision']:.4f}, Recall: {efficientnet_metrics['recall']:.4f}, F1: {efficientnet_metrics['f1_score']:.4f}")
    
    print("\nModel development completed successfully!")
