import streamlit as st
from PIL import Image
import io
import base64
import numpy as np
import matplotlib.pyplot as plt

# Fonction de simulation pour remplacer le modèle TensorFlow
def simulate_prediction(image):
    # Cette fonction simule une prédiction sans utiliser TensorFlow
    # Vous pouvez ajuster les valeurs pour démontrer différents scénarios
    # Ici, nous générons une prédiction aléatoire mais biaisée
    
    # Convertir l'image en array et extraire quelques caractéristiques simples
    img_array = np.array(image)
    avg_pixel = np.mean(img_array)
    
    # Utiliser ces caractéristiques pour générer une prédiction simulée
    # Plus la valeur moyenne des pixels est élevée, plus la probabilité d'infection est élevée
    base_prob = avg_pixel / 255.0  # Normaliser entre 0 et 1
    prediction = min(max(base_prob * 1.2, 0.1), 0.9)  # Ajuster pour éviter les extrêmes
    
    return prediction

# Interface utilisateur Streamlit
def main():
    st.set_page_config(
        page_title="Système de Détection de Poux de Mer (Version Démo)",
        page_icon="🐟",
        layout="wide"
    )
    
    # En-tête principal
    st.title("Système de Détection de Poux de Mer (Version Démo)")
    st.markdown("""
    Cette version de démonstration simule la détection de poux de mer chez les saumons.
    Téléchargez une image de saumon pour voir une analyse simulée.
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
        
        # Simuler la prédiction
        with st.spinner("Analyse en cours..."):
            prediction = simulate_prediction(image)
            is_infected = prediction > 0.5
            confidence = prediction * 100
        
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
    tab1, tab2, tab3 = st.tabs(["À propos du système", "Comment ça marche", "Version complète"])
    
    with tab1:
        st.header("À propos du Système de Détection de Poux de Mer")
        st.markdown("""
        Ce projet implémente une solution de vision par ordinateur pour détecter les infections de poux de mer dans les exploitations d'élevage de saumons.
        
        **Note** : Cette version de démonstration simule les résultats et ne contient pas le modèle d'IA complet pour des raisons de compatibilité d'hébergement.
        """)
    
    with tab2:
        st.header("Comment ça marche")
        st.markdown("""
        Dans la version complète, le système fonctionne en plusieurs étapes :
        
        1. **Prétraitement de l'image** : L'image téléchargée est redimensionnée et normalisée.
        2. **Analyse par IA** : L'image prétraitée est analysée par un modèle d'apprentissage profond.
        3. **Classification** : Le modèle détermine si l'image montre un saumon sain ou infecté.
        
        Cette version de démonstration simule ces étapes pour montrer l'interface utilisateur.
        """)
    
    with tab3:
        st.header("Accéder à la version complète")
        st.markdown("""
        La version complète de cette application avec les modèles d'IA réels est disponible sur GitHub :
        
        [https://github.com/Arak00/sea-lice-detector](https://github.com/Arak00/sea-lice-detector)
        
        Vous pouvez télécharger le code source et l'exécuter localement avec toutes les fonctionnalités.
        """)

if __name__ == "__main__":
    main()
