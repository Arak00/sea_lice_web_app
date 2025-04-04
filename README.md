Système de Détection de Poux de Mer
Aperçu du Projet
Ce projet implémente une solution de vision par ordinateur pour détecter les infections de poux de mer dans les exploitations d'élevage de saumons. Le système utilise des modèles d'apprentissage profond (ResNet50 et EfficientNet) pour analyser les images de saumons et identifier s'ils présentent des signes d'infection par des poux de mer.
Ce prototype démontre la faisabilité d'automatiser le processus de détection pour améliorer la couverture de surveillance par rapport au processus manuel actuel (qui ne contrôle qu'environ 1% des poissons toutes les 24 heures) vers un système automatisé plus complet.
Fonctionnalités
Téléchargement et analyse d'images de saumons
Détection automatique de poux de mer avec deux modèles au choix (ResNet50 ou EfficientNet)
Affichage des résultats avec niveau de confiance et visualisation
Interface utilisateur intuitive et responsive
Disponible en deux versions : Flask (interface web classique) et Streamlit (interface interactive)
Structure du Projet
sea-lice-detector/
├── sea_lice_detection/         # Dossier principal du prototype original
│   ├── data/                   # Données d'entraînement et de test
│   │   ├── processed/          # Images prétraitées
│   │   └── samples/            # Images d'exemple
│   ├── models/                 # Modèles pré-entraînés
│   ├── src/                    # Code source du prototype
│   └── README.md               # Documentation du prototype original
├── web_app.py                  # Application web Flask
├── streamlit_app.py            # Application Streamlit
├── templates/                  # Templates HTML pour Flask
└── static/                     # Fichiers statiques (CSS, JS) pour Flask
Installation
Prérequis
Python 3.10 ou supérieur
pip (gestionnaire de paquets Python)
Installation des dépendances
bash
# Installation des dépendances communes
pip install tensorflow opencv-python matplotlib scikit-learn pillow

# Pour l'application Flask
pip install flask

# Pour l'application Streamlit
pip install streamlit
Utilisation
Application Streamlit (Recommandée)
L'application Streamlit offre une interface interactive plus adaptée aux applications de data science.
bash
streamlit run streamlit_app.py
L'application s'ouvrira automatiquement dans votre navigateur web par défaut.
Application Flask
L'application Flask offre une interface web classique avec un design responsive.
bash
python web_app.py
Accédez ensuite à http://localhost:5000 dans votre navigateur web.
Fonctionnement
Le système fonctionne en plusieurs étapes :
Prétraitement de l'image : L'image téléchargée est redimensionnée à 600x250 pixels et normalisée.
Analyse par IA : L'image prétraitée est analysée par un modèle d'apprentissage profond (ResNet50 ou EfficientNet) qui a été entraîné pour reconnaître les signes d'infection par des poux de mer.
Classification : Le modèle détermine si l'image montre un saumon sain ou infecté, avec un niveau de confiance associé.
Visualisation des résultats : Les résultats sont présentés avec un code couleur (vert pour sain, rouge pour infecté) et un graphique montrant les probabilités pour chaque classe.
Limitations Actuelles
Ce prototype présente certaines limitations :
Données synthétiques : Le prototype utilise des données synthétiques qui manquent de la complexité et de la variabilité des images réelles de saumons.
Performance du modèle : La précision est inférieure à l'objectif en raison des limitations des données.
Vitesse de traitement : L'implémentation actuelle n'est pas optimisée pour le traitement vidéo en temps réel.
Améliorations Futures
Avec des données réelles et une optimisation supplémentaire, ce système pourrait :
Améliorer considérablement la couverture de surveillance (de 1% à potentiellement 100% des poissons)
Permettre une identification plus rapide des poissons infectés
Réduire la propagation des poux de mer grâce à une intervention précoce
Ajouter des capacités de traitement vidéo en temps réel
Implémenter le suivi des poissons individuels à travers les images
Ajouter la classification de la gravité des infections
Contribution
Les contributions à ce projet sont les bienvenues. N'hésitez pas à ouvrir une issue ou à soumettre une pull request.
Licence
Ce projet est distribué sous licence MIT. Voir le fichier LICENSE pour plus d'informations.
