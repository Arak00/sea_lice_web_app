import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Utiliser le backend non-interactif
from flask import Flask, render_template, request, jsonify, url_for, redirect, flash
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
from PIL import Image
import json

# Ajouter le répertoire parent au chemin pour importer config
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from sea_lice_detection.src.config import MODEL_DIR, IMG_WIDTH, IMG_HEIGHT

app = Flask(__name__)
app.secret_key = 'sea_lice_detection_secret_key'
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp'}

# Créer le dossier d'uploads s'il n'existe pas
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Variables globales pour les modèles
resnet_model = None
efficientnet_model = None

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_models():
    global resnet_model, efficientnet_model
    
    # Charger le modèle ResNet50
    resnet_path = os.path.join(MODEL_DIR, "resnet50_sea_lice_detector_best.h5")
    if not os.path.exists(resnet_path):
        resnet_path = os.path.join(MODEL_DIR, "resnet50_sea_lice_detector_final.h5")
    
    # Charger le modèle EfficientNet
    efficientnet_path = os.path.join(MODEL_DIR, "efficientnet_sea_lice_detector_best.h5")
    if not os.path.exists(efficientnet_path):
        efficientnet_path = os.path.join(MODEL_DIR, "efficientnet_sea_lice_detector_final.h5")
    
    try:
        if os.path.exists(resnet_path):
            resnet_model = load_model(resnet_path)
            print(f"Modèle ResNet50 chargé depuis {resnet_path}")
        else:
            print(f"Fichier de modèle ResNet50 non trouvé à {resnet_path}")
            resnet_model = create_demo_model()
            
        if os.path.exists(efficientnet_path):
            efficientnet_model = load_model(efficientnet_path)
            print(f"Modèle EfficientNet chargé depuis {efficientnet_path}")
        else:
            print(f"Fichier de modèle EfficientNet non trouvé à {efficientnet_path}")
            efficientnet_model = create_demo_model()
            
    except Exception as e:
        print(f"Erreur lors du chargement des modèles: {e}")
        resnet_model = create_demo_model()
        efficientnet_model = create_demo_model()

def create_demo_model():
    """Créer un modèle simple à des fins de démonstration si le modèle entraîné n'est pas disponible"""
    inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print("Modèle de démonstration créé")
    return model

def process_image(image_path):
    # Lire l'image
    img = cv2.imread(image_path)
    
    # Convertir BGR en RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Redimensionner l'image aux dimensions requises
    processed_img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    
    return processed_img

def predict_sea_lice(image_path, model_type='ResNet50'):
    # Traiter l'image
    processed_img = process_image(image_path)
    
    # Normaliser l'image
    img = processed_img.astype('float32') / 255.0
    
    # Étendre les dimensions pour correspondre à la forme d'entrée du modèle
    img = np.expand_dims(img, axis=0)
    
    # Faire la prédiction
    if model_type == 'ResNet50':
        prediction = resnet_model.predict(img)[0][0]
    else:
        prediction = efficientnet_model.predict(img)[0][0]
    
    # Déterminer le résultat
    confidence = prediction * 100
    is_infected = prediction > 0.5
    
    # Générer la visualisation
    fig, ax = plt.subplots(figsize=(8, 4))
    categories = ['Sain', 'Infecté']
    values = [1 - prediction, prediction]
    colors = ['#4CAF50', '#f44336']
    
    ax.bar(categories, values, color=colors)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Probabilité')
    ax.set_title('Probabilité de classification')
    
    # Sauvegarder le graphique dans un buffer
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Convertir le buffer en base64 pour l'affichage HTML
    plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    
    # Convertir l'image originale en base64 pour l'affichage HTML
    img_pil = Image.fromarray(processed_img)
    img_buffer = BytesIO()
    img_pil.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    img_data = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    
    return {
        'is_infected': bool(is_infected),
        'confidence': float(confidence),
        'plot_data': plot_data,
        'image_data': img_data
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('Aucun fichier sélectionné')
        return redirect(request.url)
    
    file = request.files['file']
    model_type = request.form.get('model_type', 'ResNet50')
    
    if file.filename == '':
        flash('Aucun fichier sélectionné')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Faire la prédiction
        result = predict_sea_lice(file_path, model_type)
        
        return render_template('result.html', 
                              result=result, 
                              model_type=model_type,
                              filename=filename)
    
    flash('Type de fichier non autorisé')
    return redirect(request.url)

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    # Charger les modèles au démarrage
    load_models()
    # Démarrer l'application sur toutes les interfaces réseau
    app.run(host='0.0.0.0', port=5000, debug=True)
