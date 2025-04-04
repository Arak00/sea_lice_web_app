import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog, Button, Label, Frame, StringVar, OptionMenu
from PIL import Image, ImageTk

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import MODEL_DIR, IMG_WIDTH, IMG_HEIGHT

class SeaLiceDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sea Lice Detection System")
        self.root.geometry("1000x700")
        self.root.configure(bg="#f0f0f0")
        
        # Model selection
        self.model_var = StringVar(root)
        self.model_var.set("ResNet50")  # default value
        
        # Initialize model
        self.model = None
        self.load_selected_model()
        
        # Create UI elements
        self.create_ui()
        
        # Initialize variables
        self.current_image = None
        self.current_image_path = None
        self.processed_image = None
        
    def create_ui(self):
        # Main frame
        main_frame = Frame(self.root, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title_label = Label(main_frame, text="Sea Lice Detection System", font=("Arial", 24, "bold"), bg="#f0f0f0")
        title_label.pack(pady=(0, 20))
        
        # Control panel frame
        control_frame = Frame(main_frame, bg="#e0e0e0", bd=2, relief=tk.GROOVE)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20), pady=10)
        
        # Model selection
        model_label = Label(control_frame, text="Select Model:", font=("Arial", 12), bg="#e0e0e0")
        model_label.pack(pady=(20, 5), padx=10, anchor=tk.W)
        
        model_menu = OptionMenu(control_frame, self.model_var, "ResNet50", "EfficientNet", command=self.on_model_change)
        model_menu.config(width=15, font=("Arial", 10))
        model_menu.pack(pady=(0, 20), padx=10)
        
        # Buttons
        self.load_button = Button(control_frame, text="Load Image", command=self.load_image, 
                                 font=("Arial", 12), bg="#4CAF50", fg="white", width=15)
        self.load_button.pack(pady=10, padx=10)
        
        self.detect_button = Button(control_frame, text="Detect Sea Lice", command=self.detect_sea_lice, 
                                   font=("Arial", 12), bg="#2196F3", fg="white", width=15, state=tk.DISABLED)
        self.detect_button.pack(pady=10, padx=10)
        
        self.clear_button = Button(control_frame, text="Clear", command=self.clear_display, 
                                  font=("Arial", 12), bg="#f44336", fg="white", width=15, state=tk.DISABLED)
        self.clear_button.pack(pady=10, padx=10)
        
        # Information panel
        info_frame = Frame(control_frame, bg="#e0e0e0", bd=1, relief=tk.SUNKEN)
        info_frame.pack(fill=tk.X, padx=10, pady=(20, 10))
        
        info_label = Label(info_frame, text="About", font=("Arial", 12, "bold"), bg="#e0e0e0")
        info_label.pack(pady=(10, 5), anchor=tk.W, padx=10)
        
        info_text = Label(info_frame, text="This system detects sea lice infections in salmon using computer vision and deep learning. Upload an image to analyze it for signs of infection.", 
                         font=("Arial", 10), bg="#e0e0e0", justify=tk.LEFT, wraplength=200)
        info_text.pack(pady=(0, 10), padx=10)
        
        # Display frame
        self.display_frame = Frame(main_frame, bg="white", bd=2, relief=tk.GROOVE)
        self.display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, pady=10)
        
        # Image display
        self.image_label = Label(self.display_frame, bg="white", text="No image loaded", font=("Arial", 14))
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Results frame
        self.results_frame = Frame(self.display_frame, bg="white", height=150)
        self.results_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # Results label
        self.results_label = Label(self.results_frame, text="", font=("Arial", 12), bg="white")
        self.results_label.pack(pady=10)
        
        # Confidence bar
        self.confidence_frame = Frame(self.results_frame, bg="white")
        self.confidence_frame.pack(fill=tk.X, padx=20, pady=(0, 10))
        
        self.confidence_label = Label(self.confidence_frame, text="Confidence: ", font=("Arial", 12), bg="white")
        self.confidence_label.pack(side=tk.LEFT, padx=(0, 10))
        
        self.confidence_bar_frame = Frame(self.confidence_frame, bg="#e0e0e0", height=20, width=400)
        self.confidence_bar_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.confidence_bar = Frame(self.confidence_bar_frame, bg="#4CAF50", height=20, width=0)
        self.confidence_bar.place(x=0, y=0, height=20, width=0)
        
        self.confidence_value = Label(self.confidence_frame, text="0%", font=("Arial", 12), bg="white")
        self.confidence_value.pack(side=tk.LEFT, padx=(10, 0))
    
    def load_selected_model(self):
        try:
            if self.model_var.get() == "ResNet50":
                model_path = os.path.join(MODEL_DIR, "resnet50_sea_lice_detector_best.h5")
                if not os.path.exists(model_path):
                    model_path = os.path.join(MODEL_DIR, "resnet50_sea_lice_detector_final.h5")
            else:
                model_path = os.path.join(MODEL_DIR, "efficientnet_sea_lice_detector_best.h5")
                if not os.path.exists(model_path):
                    model_path = os.path.join(MODEL_DIR, "efficientnet_sea_lice_detector_final.h5")
            
            if os.path.exists(model_path):
                self.model = load_model(model_path)
                print(f"Model loaded from {model_path}")
            else:
                print(f"Model file not found at {model_path}")
                # If model file doesn't exist, create a simple model for demonstration
                self.create_demo_model()
        except Exception as e:
            print(f"Error loading model: {e}")
            # If there's an error loading the model, create a simple model for demonstration
            self.create_demo_model()
    
    def create_demo_model(self):
        """Create a simple model for demonstration purposes if the trained model is not available"""
        inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
        x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        print("Demo model created for demonstration purposes")
    
    def on_model_change(self, *args):
        self.load_selected_model()
        if self.current_image is not None:
            self.detect_sea_lice()
    
    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if file_path:
            self.current_image_path = file_path
            self.process_image(file_path)
            self.display_image(self.processed_image)
            self.detect_button.config(state=tk.NORMAL)
            self.clear_button.config(state=tk.NORMAL)
    
    def process_image(self, image_path):
        # Read image
        img = cv2.imread(image_path)
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize image to the required dimensions
        self.processed_image = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        
        # Store original image for display
        self.current_image = Image.open(image_path)
        
        # Resize for display if too large
        self.current_image.thumbnail((500, 500))
    
    def display_image(self, img):
        if isinstance(img, np.ndarray):
            # Convert numpy array to PIL Image
            img_pil = Image.fromarray(img)
        else:
            img_pil = img
        
        # Convert PIL image to PhotoImage
        img_tk = ImageTk.PhotoImage(img_pil)
        
        # Update image label
        self.image_label.config(image=img_tk, text="")
        self.image_label.image = img_tk  # Keep a reference to prevent garbage collection
    
    def detect_sea_lice(self):
        if self.processed_image is None or self.model is None:
            return
        
        # Normalize image
        img = self.processed_image.astype('float32') / 255.0
        
        # Expand dimensions to match model input shape
        img = np.expand_dims(img, axis=0)
        
        # Make prediction
        prediction = self.model.predict(img)[0][0]
        
        # Update UI with results
        self.display_results(prediction)
    
    def display_results(self, prediction):
        confidence = prediction * 100
        
        if prediction > 0.5:
            result_text = f"Result: INFECTED (Sea Lice Detected)"
            color = "#f44336"  # Red for infected
        else:
            result_text = f"Result: HEALTHY (No Sea Lice Detected)"
            color = "#4CAF50"  # Green for healthy
        
        # Update results label
        self.results_label.config(text=result_text, fg=color, font=("Arial", 14, "bold"))
        
        # Update confidence bar
        bar_width = int(400 * (confidence / 100))
        self.confidence_bar.place(x=0, y=0, height=20, width=bar_width)
        self.confidence_bar.config(bg=color)
        
        # Update confidence value
        self.confidence_value.config(text=f"{confidence:.1f}%")
        
        # Generate and display visualization
        self.generate_visualization(prediction)
    
    def generate_visualization(self, prediction):
        # Create a figure for visualization
        fig, ax = plt.subplots(figsize=(5, 2))
        
        # Create a bar chart
        categories = ['Healthy', 'Infected']
        values = [1 - prediction, prediction]
        colors = ['#4CAF50', '#f44336']
        
        ax.bar(categories, values, color=colors)
        ax.set_ylim(0, 1)
        ax.set_ylabel('Probability')
        ax.set_title('Classification Probability')
        
        # Add the figure to the UI
        for widget in self.results_frame.winfo_children():
            if isinstance(widget, FigureCanvasTkAgg):
                widget.get_tk_widget().destroy()
        
        canvas = FigureCanvasTkAgg(fig, master=self.results_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.X, expand=True)
    
    def clear_display(self):
        # Clear image display
        self.image_label.config(image="", text="No image loaded")
        
        # Clear results
        self.results_label.config(text="")
        self.confidence_bar.place(x=0, y=0, height=20, width=0)
        self.confidence_value.config(text="0%")
        
        # Clear visualization
        for widget in self.results_frame.winfo_children():
            if isinstance(widget, FigureCanvasTkAgg):
                widget.get_tk_widget().destroy()
        
        # Reset variables
        self.current_image = None
        self.current_image_path = None
        self.processed_image = None
        
        # Disable buttons
        self.detect_button.config(state=tk.DISABLED)
        self.clear_button.config(state=tk.DISABLED)

def main():
    root = tk.Tk()
    app = SeaLiceDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
