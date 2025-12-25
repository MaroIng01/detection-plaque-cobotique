from roboflow import Roboflow
import supervision as sv
import cv2
import google.generativeai as genai
from PIL import Image
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ------------------------------------
# 0. CONFIG YOLO (VEHICLE DETECTION)
# ------------------------------------
# Charge le modèle YOLOv8 nano (rapide et léger)
print("⏳ Chargement du modèle de détection de véhicule...")
vehicle_model = YOLO("yolov8n.pt")


import os
from dotenv import load_dotenv

load_dotenv()

# ------------------------------------
# 1. CONFIG GEMINI
# ------------------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("❌ Erreur : GOOGLE_API_KEY introuvable dans le fichier .env")
    exit()

genai.configure(api_key=GOOGLE_API_KEY)
gemini = genai.GenerativeModel("models/gemini-2.5-flash")

# ------------------------------------
# 2. CONFIG ROBOFLOW
# ------------------------------------
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
if not ROBOFLOW_API_KEY:
    print("❌ Erreur : ROBOFLOW_API_KEY introuvable dans le fichier .env")
    exit()

rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace().project("licence-plate-detector-g7sp1")
model = project.version(1).model

IMAGE_PATH = "voiture/voiture6.jpg"

# ------------------------------------
# 3. PREDICTION VÉHICULE (YOLO)
# ------------------------------------
print(f"🚗 Vérification de la présence d'un véhicule dans {IMAGE_PATH}...")
# Baisse du seuil de confiance à 0.05 pour voir si la moto est détectée même faiblement
vehicle_results = vehicle_model(IMAGE_PATH, verbose=False, conf=0.05)

# Classes COCO : 2=car, 3=motorcycle, 5=bus, 7=truck
VEHICLE_CLASSES = [2, 3, 5, 7]

# Filtrer uniquement les boîtes correspondant aux véhicules
vehicle_boxes = []
for box in vehicle_results[0].boxes:
    cls = int(box.cls[0])
    conf = float(box.conf[0])
    if cls in VEHICLE_CLASSES:
        vehicle_boxes.append((cls, conf))

if not vehicle_boxes:
    print("❌ Aucun véhicule autorisé détecté.")
    exit()

# Trouver le véhicule avec la plus haute confiance
best_vehicle = max(vehicle_boxes, key=lambda x: x[1])
best_cls, best_conf = best_vehicle
best_name = vehicle_model.names[best_cls]

# Optionnel : Afficher le véhicule trouvé
# print(f"📊 Véhicule principal détecté : {best_name} (confiance {best_conf:.2f})")

if best_cls == 2: # 2 = Car
    print("✅ Voiture détectée ! Passage à la détection de plaque...")
else:
    print(f"⛔ Accès refusé : Le véhicule principal détecté est un(e) {best_name}.")
    exit()

# ------------------------------------
# 4. PREDICTION ROBOFLOW (PLAQUE)
# ------------------------------------
result = model.predict(IMAGE_PATH, confidence=40, overlap=30).json()

if len(result["predictions"]) == 0:
    print("❌ Aucune plaque détectée.")
    exit()

# Prendre la première box détectée
pred = result["predictions"][0]
x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]

# Convertir en coordonnées de crop
x1 = int(x - w/2)
y1 = int(y - h/2)
x2 = int(x + w/2)
y2 = int(y + h/2)

print("Bounding box détectée :", x1, y1, x2, y2)

# ------------------------------------
# 5. CROP DE LA PLAQUE
# ------------------------------------
img_cv = cv2.imread(IMAGE_PATH)
img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

plate_crop = img_cv[y1:y2, x1:x2]
plate_crop_rgb = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)

CROPPED_PATH = "plate_crop.jpg"
cv2.imwrite(CROPPED_PATH, plate_crop)

print("✔ Plaque cropée enregistrée :", CROPPED_PATH)

# ------------------------------------
# 6. AFFICHAGE DES IMAGES
# ------------------------------------

# Dessiner la bounding box sur l’image originale
img_box = img_rgb.copy()
cv2.rectangle(img_box, (x1, y1), (x2, y2), (255,0,0), 3)

plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.title("Image Originale + Détection")
plt.imshow(img_box)
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Plaque Cropée")
plt.imshow(plate_crop_rgb)
plt.axis("off")

plt.pause(3)
plt.close()

# ------------------------------------
# 7. OCR AVEC GEMINI
# ------------------------------------
plate_pil = Image.fromarray(plate_crop_rgb)

prompt = """
Tu vas lire une plaque d'immatriculation marocaine.
Elle contient généralement 3 parties :
1. Le numéro principal (1 à 6 chiffres)
2. Une lettre arabe
3. Le code régional (1 à 2 chiffres)

Exemples :
- 123456 أ 34
- 8765 و 1

Donne-moi EXACTEMENT le texte de la plaque, sans interprétation.
Seulement le texte, pas d'explication.
"""
response = gemini.generate_content([prompt, plate_pil])

import sqlite3

# ------------------------------------
# 8. VÉRIFICATION BASE DE DONNÉES
# ------------------------------------
print("------------------------------------")
ocr_text = response.text.strip()
print(f"� Plaque lue : {ocr_text}")

# Normalisation pour la comparaison (supprimer espaces, tirets, retours ligne)
# Ex: "15555 ه 1" devient "15555ه1"
normalized_ocr = ocr_text.replace(" ", "").replace("-", "").replace("\n", "").replace("\r", "")

# Connexion BDD
conn = sqlite3.connect("whitelist.db")
cursor = conn.cursor()

# Recherche exacte sur la version normalisée
cursor.execute("SELECT owner, plate_number FROM authorized_vehicles WHERE normalized_plate = ?", (normalized_ocr,))
result = cursor.fetchone()

conn.close()

if result:
    owner_name, plate_db = result
    print(f"✅ ACCÈS AUTORISÉ ! Bienvenue {owner_name}.")
    print(f"   (Plaque identifiée : {plate_db})")
else:
    print("⛔ ACCÈS REFUSÉ : Plaque inconnue ou non autorisée.")
    print(f"   (Lue : '{normalized_ocr}' vs attendu en base)")
