from roboflow import Roboflow
import supervision as sv
import cv2
import google.generativeai as genai
from PIL import Image
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os
import sqlite3
import time
from dotenv import load_dotenv

# ------------------------------------
# 0. CONFIG GPIO (LEDS)
# ------------------------------------
print("🔌 Configuration des GPIO pour Raspberry Pi...")
try:
    from gpiozero import LED
    # Configuration des LEDs sur les pins GPIO 17 (Vert) et 27 (Rouge)
    green_led = LED(17)
    red_led = LED(27)
    
    def access_granted():
        print("🟢 LED VERTE ALLUMÉE - Accès Autorisé")
        green_led.on()
        red_led.off()
        
    def access_denied():
        print("🔴 LED ROUGE ALLUMÉE - Accès Refusé")
        red_led.on()
        green_led.off()
        
    def cleanup_leds():
        green_led.off()
        red_led.off()

except ImportError:
    print("⚠️ Attention: 'gpiozero' n'est pas installé ou non supporté (Windows?). Mode simulation.")
    # Classes Dummy pour ne pas faire planter le script sur Windows
    class MockLED:
        def on(self): pass
        def off(self): pass
    green_led = MockLED()
    red_led = MockLED()
    
    def access_granted():
        print("🟢 [SIMULATION] LED VERTE ALLUMÉE - Accès Autorisé")
        
    def access_denied():
        print("🔴 [SIMULATION] LED ROUGE ALLUMÉE - Accès Refusé")

    def cleanup_leds():
        pass


# Initialisation : tout éteint
cleanup_leds()

# ------------------------------------
# 1. CONFIG ENV & MODELS
# ------------------------------------
load_dotenv()

# Config Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("❌ Erreur : GOOGLE_API_KEY introuvable dans le fichier .env")
    access_denied()
    exit()

genai.configure(api_key=GOOGLE_API_KEY)
# Utilisation du modèle flash pour la rapidité
gemini = genai.GenerativeModel("models/gemini-1.5-flash") # Ou 2.5-flash si disponible

# Config Roboflow
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
if not ROBOFLOW_API_KEY:
    print("❌ Erreur : ROBOFLOW_API_KEY introuvable dans le fichier .env")
    access_denied()
    exit()

rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace().project("licence-plate-detector-g7sp1")
plate_model = project.version(1).model

# Config YOLO
print("⏳ Chargement du modèle de détection de véhicule...")
vehicle_model = YOLO("yolov8n.pt")


# ------------------------------------
# 2. RUNTIME LOGIC
# ------------------------------------

def process_image(image_path):
    print(f"\n📸 Traitement de l'image : {image_path}")
    
    # ------------------------------------
    # A. DÉTECTION VÉHICULE (YOLO)
    # ------------------------------------
    # Baisse du seuil de confiance à 0.05
    vehicle_results = vehicle_model(image_path, verbose=False, conf=0.05)
    
    # Classes COCO : 2=car, 3=motorcycle, 5=bus, 7=truck
    VEHICLE_CLASSES = [2, 3, 5, 7]
    
    vehicle_boxes = []
    for box in vehicle_results[0].boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        if cls in VEHICLE_CLASSES:
            vehicle_boxes.append((cls, conf))
    
    if not vehicle_boxes:
        print("❌ Aucun véhicule autorisé détecté.")
        access_denied()
        return

    # Trouver le véhicule avec la plus haute confiance
    best_vehicle = max(vehicle_boxes, key=lambda x: x[1])
    best_cls, best_conf = best_vehicle
    best_name = vehicle_model.names[best_cls]
    
    if best_cls == 2: # 2 = Car
        print("✅ Voiture détectée ! Passage à la détection de plaque...")
    else:
        print(f"⛔ Accès refusé : Le véhicule principal est un(e) {best_name}.")
        access_denied()
        return

    # ------------------------------------
    # B. DÉTECTION PLAQUE (ROBOFLOW)
    # ------------------------------------
    result = plate_model.predict(image_path, confidence=40, overlap=30).json()

    if len(result["predictions"]) == 0:
        print("❌ Aucune plaque détectée sur la voiture.")
        access_denied()
        return

    # Prendre la première box détectée
    pred = result["predictions"][0]
    x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
    
    # Crop
    x1 = int(x - w/2)
    y1 = int(y - h/2)
    x2 = int(x + w/2)
    y2 = int(y + h/2)
    
    img_cv = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    plate_crop = img_cv[y1:y2, x1:x2]
    plate_crop_rgb = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)
    
    # Sauvegarde locale pour debug
    cv2.imwrite("plate_crop_pi.jpg", plate_crop)
    
    # ------------------------------------
    # C. OCR (GEMINI)
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
    try:
        response = gemini.generate_content([prompt, plate_pil])
        ocr_text = response.text.strip()
        print(f"📝 Plaque lue (OCR) : {ocr_text}")
    except Exception as e:
        print(f"❌ Erreur Gemini : {e}")
        access_denied()
        return

    # ------------------------------------
    # D. VÉRIFICATION BDD
    # ------------------------------------
    # Normalisation
    normalized_ocr = ocr_text.replace(" ", "").replace("-", "").replace("\n", "").replace("\r", "")
    
    conn = sqlite3.connect("whitelist.db")
    cursor = conn.cursor()
    cursor.execute("SELECT owner, plate_number FROM authorized_vehicles WHERE normalized_plate = ?", (normalized_ocr,))
    db_result = cursor.fetchone()
    conn.close()
    
    if db_result:
        owner_name, plate_db = db_result
        print(f"🎉 ACCÈS AUTORISÉ ! Bienvenue {owner_name}.")
        access_granted()
    else:
        print(f"⛔ ACCÈS REFUSÉ : Plaque '{normalized_ocr}' inconnue.")
        access_denied()

# ------------------------------------
# MAIN
# ------------------------------------
if __name__ == "__main__":
    # Pour le test, on utilise l'image statique comme demandé, 
    # mais sur Pi on pourrait capturer depuis la caméra.
    image_to_test = "voiture/voiture6.jpg" 
    
    if os.path.exists(image_to_test):
        process_image(image_to_test)
        
        # Sur Raspberry Pi, on veut peut-être garder la LED allumée quelques secondes
        print("⏳ Maintien de l'état LED pour 5 secondes...")
        time.sleep(5)
        cleanup_leds()
        print("Fin du programme.")
    else:
        print(f"❌ Image introuvable : {image_to_test}")
