# 🚗 Système de Détection et d'Autorisation de Véhicules

Ce projet est un système intelligent de contrôle d'accès basé sur la vision par ordinateur et l'IA générative.

## 🌟 Fonctionnalités

1.  **Détection de Véhicule (YOLOv8)** :
    *   Vérifie d'abord si un véhicule est présent.
    *   **Filtrage Strict** : Seules les **voitures** sont acceptées. Les motos, bus et camions sont automatiquement rejetés.

2.  **Détection de Plaque (Roboflow)** :
    *   Utilise un modèle entraîné sur Roboflow pour localiser la plaque d'immatriculation.
    *   Extrait (crop) la zone de la plaque.

3.  **Lecture OCR (Google Gemini)** :
    *   Analyse l'image de la plaque avec l'IA Gemini 2.5 Flash pour en extraire le texte (Lettres arabes et chiffres).

4.  **Contrôle d'Accès (SQLite)** :
    *   Compare la plaque lue avec une base de données locale (`whitelist.db`) de véhicules autorisés.
    *   ✅ **Accès Autorisé** : Si la plaque est reconnue.
    *   ⛔ **Accès Refusé** : Si la plaque est inconnue.

## 🛠️ Installation

1.  Cloner le dépôt :
    ```bash
    git clone https://github.com/MaroIng01/detection-plaque-cobotique.git
    cd detection-plaque-cobotique
    ```

2.  Installer les dépendances :
    ```bash
    pip install roboflow supervision opencv-python google-generativeai pillow matplotlib ultralytics
    ```

3.  Configurer les clés API :
    *   Ouvrez `geminiyolo.py` et ajoutez vos clés API pour Google Gemini et Roboflow.
    *   (Optionnel : Utilisez un fichier `.env` pour plus de sécurité).

4.  Initialiser la base de données :
    ```bash
    python database_setup.py
    ```
    *   Modifiez ce script pour ajouter vos propres plaques autorisées.

## 🚀 Utilisation

1.  Placer une image de test dans un dossier (ex: `voiture/`).
2.  Modifier la variable `IMAGE_PATH` dans `geminiyolo.py`.
3.  Lancer le script :
    ```bash
    python geminiyolo.py
    ```

## 📂 Structure du Projet

*   `geminiyolo.py` : Script principal (Pipeline de détection).
*   `database_setup.py` : Script de gestion de la base de données SQLite.
*   `whitelist.db` : Base de données locale (générée automatiquement).
*   `yolov8n.pt` : Modèle YOLO (téléchargé automatiquement).

## 📝 Auteur
Projet Cobotique - Marouane Acharifi
