import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import io
import base64

# Ajouter le répertoire parent au chemin pour importer config
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from sea_lice_detection.src.config import MODEL_DIR, IMG_WIDTH, IMG_HEIGHT

# Variables globales pour les modèles
resnet_model = None
efficientnet_model = None

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
            st.sidebar.success(f"Modèle ResNet50 chargé")
        else:
            st.sidebar.warning(f"Fichier de modèle ResNet50 non trouvé")
            resnet_model = create_demo_model()
            
        if os.path.exists(efficientnet_path):
            efficientnet_model = load_model(efficientnet_path)
            st.sidebar.success(f"Modèle EfficientNet chargé")
        else:
            st.sidebar.warning(f"Fichier de modèle EfficientNet non trouvé")
            efficientnet_model = create_demo_model()
            
    except Exception as e:
        st.sidebar.error(f"Erreur lors du chargement des modèles: {e}")
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
    st.sidebar.info("Modèle de démonstration créé")
    return model

def process_image(uploaded_file):
    # Lire l'image depuis le fichier téléchargé
    image_bytes = uploaded_file.getvalue()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Convertir BGR en RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Redimensionner l'image aux dimensions requises
    processed_img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    
    return processed_img, img

def predict_sea_lice(processed_img, model_type='ResNet50'):
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
    
    return is_infected, confidence, prediction

def main():
    st.set_page_config(
        page_title="Système de Détection de Poux de Mer",
        page_icon="🐟",
        layout="wide"
    )
    
    # Charger les modèles
    load_models()
    
    # Sidebar
    st.sidebar.title("Options")
    model_type = st.sidebar.radio(
        "Sélectionner un modèle",
        ["ResNet50", "EfficientNet"]
    )
    
    # En-tête principal
    st.title("Système de Détection de Poux de Mer")
    st.markdown("""
    Ce système utilise l'intelligence artificielle pour détecter les infections de poux de mer chez les saumons.
    Téléchargez une image de saumon pour l'analyser.
    """)
    
    # Zone de téléchargement d'image
    uploaded_file = st.file_uploader("Choisir une image...", type=["jpg", "jpeg", "png", "bmp"])
    
    if uploaded_file is not None:
        # Afficher l'image originale
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Image téléchargée")
            image = Image.open(uploaded_file)
            st.image(image, caption="Image originale", use_column_width=True)
        
        # Traiter l'image et faire la prédiction
        processed_img, original_img = process_image(uploaded_file)
        
        with st.spinner(f"Analyse en cours avec le modèle {model_type}..."):
            is_infected, confidence, prediction = predict_sea_lice(processed_img, model_type)
        
        # Afficher les résultats
        with col2:
            st.subheader("Résultat de l'analyse")
            
            if is_infected:
                st.error("**INFECTÉ**: Poux de mer détectés")
            else:
                st.success("**SAIN**: Aucun poux de mer détecté")
            
            st.metric("Niveau de confiance", f"{confidence:.1f}%")
            
            # Créer un graphique de probabilité
            fig, ax = plt.subplots(figsize=(8, 4))
            categories = ['Sain', 'Infecté']
            values = [1 - prediction, prediction]
            colors = ['#4CAF50', '#f44336']
            
            ax.bar(categories, values, color=colors)
            ax.set_ylim(0, 1)
            ax.set_ylabel('Probabilité')
            ax.set_title('Probabilité de classification')
            
            st.pyplot(fig)
    
    # Onglets d'information
    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["À propos du système", "Comment ça marche", "Limitations"])
    
    with tab1:
        st.header("À propos du Système de Détection de Poux de Mer")
        st.markdown("""
        Ce projet implémente une solution de vision par ordinateur pour détecter les infections de poux de mer dans les exploitations d'élevage de saumons. 
        Le système utilise des modèles d'apprentissage profond pour analyser les images de saumons et identifier s'ils présentent des signes d'infection par des poux de mer.
        
        Ce prototype démontre la faisabilité d'automatiser le processus de détection pour améliorer la couverture de surveillance par rapport au processus manuel actuel 
        (qui ne contrôle qu'environ 1% des poissons toutes les 24 heures) vers un système automatisé plus complet.
        """)
    
    with tab2:
        st.header("Comment ça marche")
        st.markdown("""
        Le système fonctionne en plusieurs étapes :
        
        1. **Prétraitement de l'image** : L'image téléchargée est redimensionnée à 600x250 pixels et normalisée.
        
        2. **Analyse par IA** : L'image prétraitée est analysée par un modèle d'apprentissage profond (ResNet50 ou EfficientNet) 
        qui a été entraîné pour reconnaître les signes d'infection par des poux de mer.
        
        3. **Classification** : Le modèle détermine si l'image montre un saumon sain ou infecté, avec un niveau de confiance associé.
        
        4. **Visualisation des résultats** : Les résultats sont présentés avec un code couleur (vert pour sain, rouge pour infecté) 
        et un graphique montrant les probabilités pour chaque classe.
        """)
    
    with tab3:
        st.header("Limitations actuelles")
        st.markdown("""
        Ce prototype présente certaines limitations :
        
        1. **Données synthétiques** : Le prototype utilise des données synthétiques qui manquent de la complexité et de la variabilité des images réelles de saumons.
        
        2. **Performance du modèle** : La précision est inférieure à l'objectif en raison des limitations des données.
        
        3. **Vitesse de traitement** : L'implémentation actuelle n'est pas optimisée pour le traitement vidéo en temps réel.
        
        Avec des données réelles et une optimisation supplémentaire, ce système pourrait améliorer considérablement la couverture de surveillance, 
        passant de 1% actuellement à potentiellement 100% des poissons, permettant une identification plus rapide des poissons infectés 
        et réduisant la propagation des poux de mer grâce à une intervention précoce.
        """)

if __name__ == "__main__":
    main()
